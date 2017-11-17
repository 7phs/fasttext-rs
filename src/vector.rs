use libc::{c_int, c_float, c_void};

#[repr(C)]
pub(crate) struct WrapperWordVector(c_void);

extern "C" {
    fn VEC_Len(wrapper: *const WrapperWordVector) -> c_int;
    fn VEC_GetData(wrapper: *const WrapperWordVector) -> *const c_float;
    fn VEC_Release(wrapper: *const WrapperWordVector);
}

pub struct Vector<'a> {
    wrapper: *const WrapperWordVector,
    data: &'a [f32],
}

impl<'a> Vector<'a> {
    pub(crate) unsafe fn new(wrapper: *const WrapperWordVector) -> Vector<'a> {
        let len = VEC_Len(wrapper) as usize;
        let raw_data = VEC_GetData(wrapper);
        let data = ::std::slice::from_raw_parts(raw_data, len);

        Vector {
            wrapper,
            data,
        }
    }

    pub fn as_slice(&self) -> &'a [f32] {
        self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a> Drop for Vector<'a> {
    fn drop(&mut self) {
        unsafe {
            VEC_Release(self.wrapper);
        }
    }
}

#[cfg(test)]
mod testing {
    use super::*;
    use ::libc::c_short;

    extern "C" {
        fn test_VEC_New(empty: c_short, sz: c_int) -> *const WrapperWordVector;
    }

    #[test]
    fn test_vector_new() {
        let sz: i32 = 10;

        {
            let vec = unsafe { Vector::new(test_VEC_New(true as c_short, 0)) };
            assert_eq!(vec.is_empty(), true);
        }

        {
            let vec = unsafe { Vector::new(test_VEC_New(false as c_short, sz as c_int)) };
            assert_eq!(vec.is_empty(), false);

            assert_eq!(vec.as_slice(), &[1f32, 2.019136, 3.0383742, 4.0576124, 5.07685, 6.096088, 7.115326, 8.134564, 9.153802, 10.17304]);
        }
    }
}