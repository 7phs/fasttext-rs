extern crate cc;

fn main() {
    cc::Build::new()
        .flag("-std=c++11")
        .cpp(true)
        .file("src/fasttext/src/args.cc")
        .file("src/fasttext/src/dictionary.cc")
        .file("src/fasttext/src/productquantizer.cc")
        .file("src/fasttext/src/matrix.cc")
        .file("src/fasttext/src/qmatrix.cc")
        .file("src/fasttext/src/vector.cc")
        .file("src/fasttext/src/model.cc")
        .file("src/fasttext/src/utils.cc")
        .file("src/fasttext/src/fasttext.cc")
        .file("src/fasttext_wrapper.cc")
        .warnings(false)
        .compile("fasttext");   // outputs `libfasttext.a`
}