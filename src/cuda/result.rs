use std::{
    ffi::{c_uint, c_void},
    mem::{size_of, MaybeUninit},
};

use super::sys;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaError(sys::CUresult); //tuple struct with one field accessed as self.0

impl sys::CUresult {
    pub fn result(self) -> Result<(), CudaError> {
        match self {
            sys::CUresult::CUDA_SUCCESS => Ok(()),
            _ => Err(CudaError(self)),
        }
    }
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{:?}", self))
    }
}

impl std::error::Error for CudaError {}

pub fn init() -> Result<(), CudaError> {
    unsafe { sys::cuInit(0).result() }
}

//this core pattern will be used everywhere. CUresult is the C enum type returned by
//every CUDA call. We add a .result() method directly on it that converts CUDA_SUCCESS -> Ok(())
//anything else -> Err(CudaError(...)). The CudaError newtype wraps the raw enum so we can
//implement Display and Error on it.

pub mod device {
    use super::{sys, CudaError};
    use std::mem::MaybeUninit;

    pub fn get(ordinal: std::ffi::c_int) -> Result<sys::CUdevice, CudaError> {
        let mut dev: sys::CUdevice = 0;
        unsafe {sys::cuDeviceGet((&mut dev) as *mut sys::CUdevice, ordinal).result()? }
        //&mut dev is a rust reference its not a raw C pointer. we need to cast it to a raw C pointer
        Ok(dev)
    }

    pub unsafe fn primary_ctx_retain(dev: sys::CUdevice) -> Result<sys::CUcontext, CudaError> {
        let mut ctx = MaybeUninit::uninit();
        sys::cuDevicePrimaryCtxRetain(ctx.as_mut_ptr(), dev).result()?;
        Ok(ctx.assume_init())
    }

    pub unsafe fn primary_ctx_release(dev: sys::CUdevice) -> Result<(), CudaError> {
        sys::cuDevicePrimaryCtxRelease_v2(dev).result()
    }

}

pub mod ctx {
    use super::{sys, CudaError};

    pub unsafe fn set_current(ctx: sys::CUcontext) -> Result<(), CudaError> {
        sys::cuCtxSetCurrent(ctx).result()
    }
}


pub mod stream {
    use super::{sys, CudaError};
    use std::mem::MaybeUninit;

    pub use sys::CUstream_flags;

    pub fn null() -> sys::CUstream {
        std::ptr::null_mut()
    }

    pub fn create(flags: CUstream_flags) -> Result<sys::CUstream, CudaError> {
        let mut stream = MaybeUninit::uninit();
        unsafe {
            sys::cuStreamCreate(stream.as_mut_ptr(), flags as u32).result()?;
            Ok(stream.assume_init())
        }
    }

    pub unsafe fn synchronize(stream: sys::CUstream) -> Result<(), CudaError> {
        sys::cuStreamSynchronize(stream).result()
    }

    pub unsafe fn destroy(stream: sys::CUstream) -> Result<(), CudaError> {
        sys::cuStreamDestroy_v2(stream).result()
    }
}



//memory allocation and transfer
pub unsafe fn malloc<T>() -> Result<sys::CUdeviceptr, CudaError> {
    let bytesize = size_of::<T>();
    let mut dev_ptr = MaybeUninit::uninit();
    unsafe {
        sys::cuMemAlloc_v2(dev_ptr.as_mut_ptr(), bytesize).result()?;
        Ok(dev_ptr.assume_init())
    }
}

pub unsafe fn malloc_async<T>(stream: sys::CUstream) -> Result<sys::CUdeviceptr, CudaError> {
    let bytesize = size_of::<T>();
    let mut dev_ptr = MaybeUninit::uninit();
    unsafe {
        sys::cuMemAllocAsync(dev_ptr.as_mut_ptr(), bytesize, stream).result()?;
        Ok(dev_ptr.assume_init())
    }
}

pub unsafe fn free(dptr: sys::CUdeviceptr) -> Result<(), CudaError> {
    sys::cuMemFree_v2(dptr).result()
}

pub unsafe fn free_async (dptr: sys::CUdeviceptr, stream: sys::CUstream) -> Result<(), CudaError> {
    sys::cuMemFreeAsync(dptr, stream).result()
}


//Memset

pub unsafe fn memset_d8<T>(dptr: sys::CUdeviceptr, uc: std::ffi::c_uchar) -> Result<(), CudaError> {
    sys::cuMemsetD8_v2(dptr, uc, size_of::<T>()).result()
}

pub unsafe fn memset_d8_async<T>(
    dptr: sys::CUdeviceptr, 
    uc: std::ffi::c_uchar,
    stream: sys::CUstream,
) -> Result<(), CudaError> {
    sys::cuMemsetD8Async(dptr, uc, size_of::<T>(), stream).result()
}

//host <-> device memory copy
pub unsafe fn memcpy_htod<T>(dst: sys::CUdeviceptr, src: &T) -> Result<(), CudaError> {
    sys::cuMemcpyHtoD_v2(dst, src as *const T as *const _, size_of::<T>()).result()
    //from rust reference to raw c pointer. from raw c pointer to generic void pointer. void pointer in c refers to "any type of data"
}

pub unsafe fn memcpy_htod_async<T>(
    dst: sys::CUdeviceptr,
    src: &T,
    stream: sys::CUstream,
) -> Result<(), CudaError> {
    sys::cuMemcpyHtoDAsync_v2(dst, src as *const T as *const _, size_of::<T>(), stream).result()
}

pub unsafe fn memcpy_dtoh<T>(dst: &mut T, src: sys::CUdeviceptr) -> Result<(), CudaError> {
    sys::cuMemcpyDtoH_v2(dst as *mut T as *mut _, src, size_of::<T>()).result()
}

pub unsafe fn memcpy_dtoh_async<T>(
    dst: &mut T,
    src: sys::CUdeviceptr,
    stream: sys::CUstream,
) -> Result<(), CudaError> {
    sys::cuMemcpyDtoHAsync_v2(dst as *mut T as *mut _, src, size_of::<T>(), stream).result()
}