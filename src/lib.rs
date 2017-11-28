#![feature(test)]
#![feature(unique)]
#![feature(plugin)]

extern crate libc;
extern crate test;
extern crate wordvector as wordvector_base;

mod testing;
mod string;

pub mod dictionary;
pub mod fasttext;
pub mod predict;
pub mod vector;
pub mod wordvector;