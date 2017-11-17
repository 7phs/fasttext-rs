use libc::c_uint;
use std::os::raw::c_char;

#[repr(C)]
pub(crate) struct WrapperString {
    str: *mut c_char,
    len: c_uint,
    cap: c_uint,
}

impl WrapperString {
    pub(crate) unsafe fn new(buf: &Vec<u8>) -> WrapperString {
        WrapperString {
            str: buf.as_ptr() as *mut c_char,
            len: 0,
            cap: buf.len() as c_uint,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len as usize
    }
}
