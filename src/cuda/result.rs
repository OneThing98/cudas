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