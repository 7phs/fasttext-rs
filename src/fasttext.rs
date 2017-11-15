use libc::{c_int, c_uint, c_float};
use std::path::Path;
use std::os::raw::c_char;
use std::str;
use vector::{WrapperWordVector, Vector};

pub const RES_OK: ResSuccess = ResSuccess(0);

const MAX_WORD_LEN: usize = 256;

#[derive(Debug, Clone, Copy)]
pub struct ResSuccess(i32);

#[derive(Debug, Clone, Copy)]
pub enum Err {
    ResErrorNotOpen,
    RerErrorWrongModel,
    ResErrorModelNotInit,
    ResErrorExecution,
}

#[repr(C)]
struct WrapperString {
    str: *mut c_char,
    len: c_uint,
    cap: c_uint,
}

impl WrapperString {
    unsafe fn new(buf: &Vec<u8>) -> WrapperString {
        WrapperString {
            str: buf.as_ptr() as *mut c_char,
            len: 0,
            cap: buf.len() as c_uint,
        }
    }
}

#[repr(C)]
struct WrapperDictionary;

extern "C" {
    fn DICT_Find(wrapper: *const WrapperDictionary, word: *const c_char) -> c_int;
    fn DICT_GetWord(wrapper: *const WrapperDictionary, index: c_int, word: *mut WrapperString);
    fn DICT_WordsCount(wrapper: *const WrapperDictionary) -> c_int;
}

#[derive(Debug)]
pub struct Dictionary(*const WrapperDictionary);

impl Dictionary{
    pub fn find(&self, word: &str) -> Option<i32> {
        let index = unsafe { DICT_Find(self.0, word.as_ptr() as *const c_char) };
        if index >= 0 {
            Some(index)
        } else {
            None
        }
    }

    pub fn get_word(&self, index: i32) -> Option<String> {
        let mut c_word = vec![0u8; MAX_WORD_LEN];

        let len = unsafe {
            let mut wrap_word = WrapperString::new(&c_word);
            DICT_GetWord(self.0, index as c_int, &mut wrap_word);
            wrap_word.len
        } as usize;

        c_word.resize(len, 0u8);

        let word = String::from_utf8(c_word).unwrap_or_default();

        if !word.is_empty() {
            Some(word)
        } else {
            None
        }
    }

    pub fn words_count(&self) -> i32 {
        unsafe {
            DICT_WordsCount(self.0)
        }
    }
}

#[repr(C)]
struct WrapperFastText;

extern "C" {
    fn NewFastText() -> *mut WrapperFastText;
    fn FT_LoadModel(wrapper: *mut WrapperFastText, model_path: *const c_char) -> c_int;
    fn FT_LoadVectors(wrapper: *mut WrapperFastText, vectors_path: *const c_char) -> c_int;
    fn FT_GetDictionary(wrapper: *const WrapperFastText) -> *const WrapperDictionary;
    fn FT_GetWordVector(wrapper: *const WrapperFastText, word: *const c_char) -> *mut WrapperWordVector;
    fn FT_Release(wrapper: *mut WrapperFastText);
}

pub struct FastText(*mut WrapperFastText);

impl Drop for FastText {
    fn drop(&mut self) {
        unsafe {
            FT_Release(self.0)
        }
    }
}

impl FastText {
    pub fn new() -> FastText {
        unsafe {
            FastText(NewFastText())
        }
    }

    pub fn load_model(&mut self, model_path: &Path) -> Result<ResSuccess, Err> {
        unsafe {
            match FT_LoadModel(self.0, model_path.to_str().unwrap().as_ptr() as *const c_char) {
                0 => Ok(RES_OK),
                _ => Err(Err::ResErrorNotOpen),
            }
        }
    }

    pub fn load_vectors(&mut self, vectors_path: &Path) -> Result<ResSuccess, Err> {
        unsafe {
            match FT_LoadVectors(self.0, vectors_path.to_str().unwrap().as_ptr() as *const c_char) {
                0 => Ok(RES_OK),
                _ => Err(Err::ResErrorNotOpen),
            }
        }
    }

    pub fn get_dictionary(&self) -> Dictionary {
        unsafe {
            Dictionary(FT_GetDictionary(self.0))
        }
    }

    pub fn word_to_vector(&self, word: &str) -> Option<Vector> {
        let vec = unsafe { Vector::new(FT_GetWordVector(self.0, word.as_ptr() as *const c_char)) };

        if !vec.is_empty() {
            Some(vec)
        } else {
            None
        }
    }
}
