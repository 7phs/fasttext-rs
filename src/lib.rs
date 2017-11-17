#![feature(test)]
#![feature(unique)]
#![feature(plugin)]

extern crate libc;
extern crate test;

mod testing;
mod string;

pub mod dictionary;
pub mod fasttext;
pub mod predict;
pub mod vector;