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
