use std::path::Path;
use std::os::raw::c_char;
use std::os::unix::ffi::OsStrExt;
use std::str;
use std::ffi::CString;
use libc::{c_int, c_void};

use dictionary::WrapperDictionary;
use dictionary::Dictionary;
use predict::{WrapperPredictResult, Predict};
use vector::{WrapperWordVector, Vector};

pub const RES_OK: ResSuccess = ResSuccess(0);

#[derive(Debug, Clone, Copy)]
pub struct ResSuccess(i32);

#[derive(Debug, Clone, Copy)]
pub enum Err {
    ResErrorNotOpen,
    RerErrorWrongModel,
    ResErrorModelNotInit,
    ResErrorExecution,
}

unsafe fn to_ptr_const_char(path: &Path) -> CString {
    CString::new(path.as_os_str().as_bytes()).unwrap_or_default()
}

#[repr(C)]
pub(crate) struct FastTextWrapper(*mut c_void);

extern "C" {
    fn NewFastText() -> *mut c_void;
    fn FT_LoadModel(wrapper: *mut c_void, model_path: *const c_char) -> c_int;
    fn FT_LoadVectors(wrapper: *mut c_void, vectors_path: *const c_char) -> c_int;
    fn FT_GetDictionary(wrapper: *const c_void) -> *const WrapperDictionary;
    fn FT_GetWordVector(wrapper: *const c_void, word: *const c_char) -> *mut WrapperWordVector;
    fn FT_GetSentenceVector(wrapper: *const c_void, text: *const c_char) -> *mut WrapperWordVector;
    fn FT_Predict(wrapper: *const c_void, text: *const c_char, count: c_int) -> *const WrapperPredictResult;
    fn FT_Release(wrapper: *mut c_void);
}

impl Default for FastTextWrapper {
    fn default() -> FastTextWrapper {
        unsafe {
            FastTextWrapper(NewFastText())
        }
    }
}

impl Drop for FastTextWrapper {
    fn drop(&mut self) {
        unsafe {
            FT_Release(self.0)
        }
    }
}

impl FastTextWrapper {
    pub(crate) fn load_model(&mut self, model_path: &Path) -> Result<ResSuccess, Err> {
        unsafe {
            let r = FT_LoadModel(self.0, to_ptr_const_char(model_path).as_ptr() as *const c_char);
            match r {
                0 => Ok(RES_OK),
                _ => Err(Err::ResErrorNotOpen),
            }
        }
    }

    pub(crate) fn load_vectors(&mut self, vectors_path: &Path) -> Result<ResSuccess, Err> {
        unsafe {
            match FT_LoadVectors(self.0, to_ptr_const_char(vectors_path).as_ptr() as *const c_char) {
                0 => Ok(RES_OK),
                _ => Err(Err::ResErrorNotOpen),
            }
        }
    }

    pub(crate) fn get_dictionary(&self) -> Dictionary {
        Dictionary::new(unsafe { FT_GetDictionary(self.0) })
    }

    pub(crate) fn word_to_vector(&self, word: &str) -> Option<Vector> {
        let vec = unsafe { Vector::new(FT_GetWordVector(self.0, word.as_ptr() as *const c_char)) };

        if !vec.is_empty() {
            Some(vec)
        } else {
            None
        }
    }

    pub(crate) fn sentence_to_vector(&self, text: &str) -> Option<Vector> {
        let vec = unsafe { Vector::new(FT_GetSentenceVector(self.0, text.as_ptr() as *const c_char)) };

        if !vec.is_empty() {
            Some(vec)
        } else {
            None
        }
    }

    pub(crate) fn predict(&self, text: &str, count: i32) -> Result<Predict, String> {
        let predict = unsafe { Predict::new(FT_Predict(self.0, text.as_ptr() as *const c_char, count as c_int)) };

        match predict.err() {
            Ok(_) => Ok(predict),
            Err(err) => Err(err),
        }
    }
}
