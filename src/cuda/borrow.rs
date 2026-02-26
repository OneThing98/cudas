use super::result;
use super::sys;
use std::alloc::{alloc_zeroed, Layout};
use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;

pub mod prelude {
    pub use super::result::CudaError;
    pub use super::*;
}

#[derive(Debug)]
pub struct CudaDevice{
    pub(crate) cu_device: sys::CUdevice, 
    pub(crate) cu_primary_ctx: sys::CUcontext,
    pub(crate) cu_stream: sys::CUstream,
    pub(crate) loaded_modules: HashMap<&'static str, CudaModule>,
}

#[derive(Debug)]
pub struct CudaModule {
    pub(crate) cu_module: sys::CUmodule,
    pub(crate) functions: HashMap<&'static str, CudaFunction>,
}

#[derive(Debug)]
pub struct CudaFunction {
    pub(crate) cu_function: sys::CUfunction,
}

#[derive(Debug)]
pub struct InCudaMemory<'device, T> {
    pub(crate) cu_device_ptr: sys::CUdeviceptr,
    pub(crate) host_data: Option<Box<T>>,
    device: PhantomData<&'device CudaDevice>
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        for(_, module) in self.loaded_modules.drain(){
            unsafe { result::module::unload(module.cu_module) }.unwrap();
        }

        let stream = std::mem::replace(&mut self.cu_stream, std::ptr::null_mut());
        if !stream.is_null() {
            unsafe { result::stream::destroy(stream) }.unwrap();
        }

        let ctx = std::mem::replace(&mut self.cu_primary_ctx, std::ptr::null_mut());
        if !ctx.is_null() {
            unsafe { result::device::primary_ctx_release(self.cu_device) }.unwrap();
        }
    }
}

impl CudaDevice {
    pub fn new(ordinal: usize) -> Result<Self, result::CudaError> {
        result::init()?;
        let cu_device = result::device::get(ordinal as i32)?;
        let cu_primary_ctx = unsafe { result::device::primary_ctx_retain(cu_device)}?;
        unsafe { result::ctx::set_current(cu_primary_ctx) }?;
        let cu_stream = result::stream::create(result::stream::CUstream_flags::CU_STREAM_NON_BLOCKING)?;
        Ok(Self{
            cu_device,
            cu_primary_ctx,
            cu_stream,
            loaded_modules: HashMap::new(),
        })
    }

    //unsafe because it memsets all allocated memory to 0, and T may not be valid.
    pub unsafe fn alloc<T>(&self) -> Result<InCudaMemory<T>, result::CudaError> {
        let cu_device_ptr = unsafe {
            result::malloc_async::<T>(self.cu_stream)
        }?;
        unsafe {
            result::memset_d8_async::<T>(cu_device_ptr, 0, self.cu_stream)
        }?;
        Ok(InCudaMemory {
            cu_device_ptr,
            host_data: None,
            device: PhantomData,
        })
    }

    //the net effect is: data starts on the CPU heap, gets copied to GPU memory, then the CPU copy is freed. The GPU now has the only copy. 
    pub fn take<T>(&self, host_data: Box<T>) -> Result<InCudaMemory<T>, result::CudaError> {
        let cu_device_ptr = unsafe {
            result::malloc_async::<T>(self.cu_stream)
        }?;
        unsafe { result::memcpy_htod_async(cu_device_ptr, host_data.as_ref(), self.cu_stream) }?;
        Ok(InCudaMemory {
            cu_device_ptr,
            host_data: Some(host_data),
            device: PhantomData,
        })
    }

    //unsafe because of the same reason with T not being valid.
    pub fn release<T>(&self, t: InCudaMemory<T>) -> Result<Box<T>, result::CudaError> {
        let mut host_data = t.host_data.unwrap_or_else(|| {
            let layout = Layout::new::<T>();
            unsafe {
                let ptr = alloc_zeroed(layout) as *mut T;
                Box::from_raw(ptr)
            }
        });
        unsafe { result::memcpy_dtoh_async(host_data.as_mut(), t.cu_device_ptr, self.cu_stream) }?;
        self.synchronize()?;
        unsafe { result::free_async(t.cu_device_ptr, self.cu_stream) }?;
        Ok(host_data)
    }

    pub fn synchronize(&self) -> Result<(), result::CudaError> {
        unsafe { result::stream::synchronize(self.cu_stream) }
    }


}