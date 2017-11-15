#![feature(test)]
#![feature(unique)]
#![feature(plugin)]
#![plugin(clippy)]

extern crate libc;
extern crate test;

mod fasttext;
mod testing;
mod vector;