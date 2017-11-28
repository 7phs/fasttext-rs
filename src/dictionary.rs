use libc::{c_int, c_void};
use std::os::raw::c_char;
use string::WrapperString;

const MAX_WORD_LEN: usize = 256;

#[repr(C)]
pub(crate) struct WrapperDictionary(c_void);

extern "C" {
    fn DICT_Find(wrapper: *const WrapperDictionary, word: *const c_char) -> c_int;
    fn DICT_GetWord(wrapper: *const WrapperDictionary, index: c_int, word: *mut WrapperString);
    fn DICT_WordsCount(wrapper: *const WrapperDictionary) -> c_int;
}

#[derive(Debug)]
pub struct Dictionary(*const WrapperDictionary);

impl Dictionary{
    pub(crate) fn new(wrapper: *const WrapperDictionary) -> Dictionary {
        Dictionary(wrapper)
    }

    pub fn word_index(&self, word: &str) -> Option<i64> {
        let index = unsafe { DICT_Find(self.0, word.as_ptr() as *const c_char) };
        if index >= 0 {
            Some(index as i64)
        } else {
            None
        }
    }

    pub fn get_word(&self, index: i64) -> Option<String> {
        let mut c_word = vec![0u8; MAX_WORD_LEN];

        let len = unsafe {
            let mut wrap_word = WrapperString::new(&c_word);
            DICT_GetWord(self.0, index as c_int, &mut wrap_word);
            wrap_word.len()
        };

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
