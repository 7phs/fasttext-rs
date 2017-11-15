use libc::{c_int, c_float, c_short};
use std::ops::Deref;
use std::ptr::Unique;

#[repr(C)]
pub(crate) struct WrapperWordVector;

extern "C" {
    fn test_VEC_New(empty: c_short, sz: c_int) -> *mut WrapperWordVector;
    fn VEC_Size(wrapper: *mut WrapperWordVector) -> c_int;
    fn VEC_GetData(wrapper: *mut WrapperWordVector) -> *const c_float;
    fn VEC_Release(wrapper: *mut WrapperWordVector);
}

pub struct Vector {
    wrapper: *mut WrapperWordVector,
    ptr: Unique<f32>,
    len: usize,
}

impl Vector {
    pub(crate) unsafe fn new(wrapper: *mut WrapperWordVector) -> Vector {
        let data = VEC_GetData(wrapper);
        let len = VEC_Size(wrapper);

        Vector {
            wrapper: wrapper,
            ptr: Unique::new_unchecked(data as *mut _),
            len: len as usize,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Drop for Vector {
    fn drop(&mut self) {
        unsafe {
            VEC_Release(self.wrapper);
        }
    }
}

impl Deref for Vector {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        unsafe {
            ::std::slice::from_raw_parts(self.ptr.as_ptr(), self.len)
        }
    }
}

#[cfg(test)]
mod testing {
    use super::*;

    #[test]
    fn test_vector_new() {
        let sz: i32 = 10;

        {
            let mut vec = unsafe { Vector::new(test_VEC_New(true as c_short, 0)) };
            assert_eq!(vec.is_empty(), true);
        }

        {
            let mut vec = unsafe { Vector::new(test_VEC_New(false as c_short, sz as c_int)) };
            assert_eq!(vec.is_empty(), false);

            assert_eq!(vec.deref(), &[1f32, 2.019136, 3.0383742, 4.0576124, 5.07685, 6.096088, 7.115326, 8.134564, 9.153802, 10.17304]);
        }
    }
}