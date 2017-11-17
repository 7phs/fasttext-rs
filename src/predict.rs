use libc::{c_int, c_float};
use std::os::raw::c_char;
use std::ffi::CStr;

const EMPTY_ERROR: &'static str = "";
const EPS: f32 = 1e-6;

#[repr(C)]
struct WrapperPredictRecord {
    predict: c_float,
    word: *const c_char,
}

#[repr(C)]
pub(crate) struct WrapperPredictResult {
    records: *const WrapperPredictRecord,
    err: *const c_char,
}

extern "C" {
    fn PRDCT_Records(wrapper: *const WrapperPredictResult) -> *mut WrapperPredictRecord;
    fn PRDCT_Len(wrapper: *const WrapperPredictResult) -> c_int;
    fn PRDCT_Error(wrapper: *const WrapperPredictResult) -> *const c_char;
    fn PRDCT_Release(wrapper: *const WrapperPredictResult);
}

#[derive(Debug)]
pub struct PredictRecord<'a>(f32, &'a str);

impl<'a> PredictRecord<'a>{
    pub fn prediction(&self) -> f32 {
        self.0
    }

    pub fn word(&self) -> &'a str {
        self.1
    }
}

impl<'a> PartialEq for PredictRecord<'a>{
    fn eq(&self, other: &PredictRecord) -> bool {
        (self.0 - other.0).abs() < EPS && self.1 == other.1
    }
}

#[derive(Debug)]
pub struct Predict<'a> {
    wrapper: *const WrapperPredictResult,
    data: Vec<PredictRecord<'a>>,
    err: &'a str,
}

impl<'a> Predict<'a> {
    pub(crate) unsafe fn new(wrapper: *const WrapperPredictResult) -> Predict<'a> {
        let ptr = PRDCT_Error(wrapper);
        let err = if ptr.is_null() {
            EMPTY_ERROR
        } else {
            CStr::from_ptr(ptr).to_str().unwrap_or_default()
        };

        let data: Vec<PredictRecord<'a>> = if !err.is_empty() {
            vec![]
        } else {
            let records = PRDCT_Records(wrapper);
            let len = PRDCT_Len(wrapper) as usize;

            ::std::slice::from_raw_parts(records, len)
                .iter()
                .map(|rec|
                    PredictRecord::<'a>(
                        rec.predict as f32,
                        CStr::from_ptr(rec.word).to_str().unwrap_or_default()
                    )
                )
                .collect()
        };

        Predict {
            wrapper,
            data,
            err,
        }
    }

    pub fn as_slice(&self) -> &[PredictRecord<'a>] {
        self.data.as_slice()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn err(&self) -> Result<(), String> {
        if self.err.is_empty() {
            Ok(())
        } else {
            Err(String::from(self.err))
        }
    }
}

impl<'a> Drop for Predict<'a> {
    fn drop(&mut self) {
        unsafe {
            PRDCT_Release(self.wrapper);
        }
    }
}

#[cfg(test)]
mod testing {
    use super::*;
    use ::libc::c_short;

    impl<'a> PredictRecord<'a> {
        pub(crate) fn new(prediction: f32, word: &'a str) -> PredictRecord<'a> {
            PredictRecord(prediction, word)
        }
    }

    extern "C" {
        fn test_PRDCT_New(err: c_short, sz: c_int) -> *const WrapperPredictResult;
    }

    #[test]
    fn test_predict_new() {
        {
            let predict = unsafe { Predict::new(test_PRDCT_New(true as c_short, 0)) };

            assert_eq!(match predict.err() {
                Ok(_) => false,
                Err(_) => true,
            }, true, "check error");
            assert_eq!(predict.is_empty(), true, "check empty result");
        }

        {
            let sz: usize = 10;
            let predict = unsafe { Predict::new(test_PRDCT_New(false as c_short, sz as c_int)) };

            assert_eq!(match predict.err() {
                Ok(_) => false,
                Err(_) => true,
            }, false, "check success");
            assert_eq!(predict.is_empty(), false, "check result");
            assert_eq!(predict.len(), sz, "check result length");

            assert_eq!(
                predict.as_slice().iter()
                    .filter(|rec|
                        rec.0<0.0f32 || rec.0>1.0f32 || rec.1.len()==0
                    ).count(),
                0, "check record validation");
        }
    }
}