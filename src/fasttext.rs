use libc::{c_int, c_uchar, calloc, memcpy, free, strnlen, c_void};
use std::path::Path;
use std::ffi::{CStr, CString, OsString};
use std::os::raw::c_char;
use std::error::Error;
use std::str;
use std::mem;

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

#[repr(C)]
struct WrapperDictionary;

extern "C" {
    fn DICT_Find(wrapper: *const WrapperDictionary, word: *const c_char) -> c_int;
    fn DICT_GetWord(wrapper: *const WrapperDictionary, index: c_int, word: *mut c_char, max_len: c_int) -> c_int;
    fn DICT_WordsCount(wrapper: *const WrapperDictionary) -> c_int;
}

#[derive(Debug)]
pub struct Dictionary(*const WrapperDictionary);

impl Dictionary {
    pub fn find(&self, word: &str) -> Option<i32> {
        let index = unsafe { DICT_Find(self.0, word.as_ptr() as *const c_char) };
        if index >= 0 {
            Some(index)
        } else {
            None
        }
    }

    pub fn get_word(&self, index: i32) -> Option<String> {
        let mut c_word = vec![0u8; 256];

        let len = unsafe { DICT_GetWord(self.0, index as c_int, c_word.as_ptr() as *mut c_char, c_word.len() as i32) } as usize;

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
}


#[cfg(test)]
mod testing {
    use super::*;
    use std::path::Path;

    const UNKNOWN_PATH: &str = "unknown path";
    const UNSUPERVISED_MODEL_PATH: &str = "./test-data/unsupervised_model.bin";
    const UNSUPERVISED_VECTORS_PATH: &str = "./test-data/unsupervised_model.vec";

    #[test]
    fn test_fasttext_new() {
        FastText::new();
    }

    #[test]
    fn test_fasttext_load_model() {
        let mut model = FastText::new();

        match model.load_model(Path::new(UNKNOWN_PATH)) {
            Ok(_) => assert!(false, "failed to raise an error for an unknown model path"),
            Err(_) => assert!(true),
        }

        match model.load_model(Path::new(UNSUPERVISED_MODEL_PATH)) {
            Ok(_) => assert!(true),
            Err(err) => assert!(false, err),
        }
    }

    #[test]
    fn test_fasttext_load_vectors() {
        let mut model = FastText::new();

        match model.load_model(Path::new(UNSUPERVISED_MODEL_PATH)) {
            Ok(_) => assert!(true),
            Err(err) => {
                println!("Failed to load model {:?} with error {:?}", UNSUPERVISED_MODEL_PATH, err);
                assert!(false)
            }
        }

        match model.load_vectors(Path::new(UNKNOWN_PATH)) {
            Ok(_) => {
                println!("failed to raise an error for an unknown vectors path");
                assert!(false)
            }
            Err(_) => assert!(true),
        }

        match model.load_vectors(Path::new(UNSUPERVISED_VECTORS_PATH)) {
            Ok(_) => assert!(true),
            Err(err) => {
                println!("Failed to load vectors {:?} with error {:?}", UNSUPERVISED_VECTORS_PATH, err);
                assert!(false)
            }
        }
    }

    #[test]
    fn test_fasttext_get_dictionary() {
        let mut model = FastText::new();

        match model.load_model(Path::new(UNSUPERVISED_MODEL_PATH)) {
            Ok(_) => assert!(true),
            Err(err) => {
                println!("Failed to load model {:?} with error {:?}", UNSUPERVISED_MODEL_PATH, err);
                assert!(false)
            }
        }

        match model.load_vectors(Path::new(UNSUPERVISED_VECTORS_PATH)) {
            Ok(_) => assert!(true),
            Err(err) => {
                println!("Failed to load vectors {:?} with error {:?}", UNSUPERVISED_VECTORS_PATH, err);
                assert!(false)
            }
        }

        let dict = model.get_dictionary();

        let w_count = dict.words_count();
        if w_count <= 0 {
            println!("word count shoud greater then 0, but got {:?}", w_count);
            assert!(false)
        }

        let expected_word: &str = "златом";
        let expected_index: i32 = 22;

        match dict.find(expected_word) {
            Some(index) => assert_eq!(index, expected_index),
            None => {
                println!("failed to found word {}", expected_word);
                assert!(false)
            }
        };

        match dict.get_word(expected_index) {
            Some(word) => assert_eq!(word, expected_word),
            None => {
                println!("failed to found word by index {}", expected_index);
                assert!(false)
            }
        };
    }
}