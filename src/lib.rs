#![feature(test)]
#![feature(unique)]
#![feature(plugin)]

extern crate libc;
extern crate test;
extern crate wordvector as wordvector_base;

mod testing;
mod string;
mod fasttext;

pub mod dictionary;
pub mod predict;
pub mod vector;
pub mod wordvector;

use fasttext::FastTextWrapper;

pub struct FastText(FastTextWrapper);

impl Default for FastText {
    fn default() -> FastText {
        FastText(FastTextWrapper::default())
    }
}
