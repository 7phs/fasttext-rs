The Rust wrapper of FaceBook [Fasttext](https://github.com/facebookresearch/fastText).


# Init submodule

Add submodule:

```
git submodule add https://github.com/facebookresearch/fastText.git src/fasttext
```

First - init and update submodule.

```
git submodule update --init

git submodule foreach git pull
```

# Make test model

Make fasttext:

```
cd src/fasttext
make
```

Build an unsupervised model:

```
src/fasttext/fasttext skipgram -input unsupervised_text.txt -output test-data/unsupervised_model
```

Build a supervised model:

```
src/fasttext/fasttext supervised -input supervised_text.txt -output test-data/supervised_model
```

# Test

```bash
cargo test -- --test-threads=1
```