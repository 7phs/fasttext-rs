use std::path::Path;

use wordvector_base::WordVectorModel;

use fasttext::{Err, FastTextWrapper};
use FastText;

impl FastText {
    pub fn with_model(path: &str) -> Result<FastText, Err> {
        let model_path: String = [path, "bin"].join(".");
        let vectors_path: String = [path, "vec"].join(".");

        let mut model = FastTextWrapper::default();
        model.load_model(Path::new(&model_path))?;
        model.load_vectors(Path::new(&vectors_path))?;

        Ok(FastText(model))
    }
}

impl WordVectorModel for FastText {
    fn word_index(&self, word: &str) -> Option<i64> {
        self.0.get_dictionary().word_index(word)
    }

    fn word_to_vector(&self, word: &str) -> Option<Vec<f32>> {
        match self.0.word_to_vector(word) {
            Some(vector) => Some(vector.as_slice().to_owned()),
            None => None,
        }
    }

    fn sentence_to_vector(&self, text: &str) -> Option<Vec<f32>> {
        match self.0.word_to_vector(text) {
            Some(vector) => Some(vector.as_slice().to_owned()),
            None => None,
        }
    }
}

#[cfg(test)]
mod testing {
    use super::*;

    #[test]
    fn test_fasttextmodel_new() {
        match FastText::with_model("./test-data/unsupervised_model") {
            Ok(model) => assert!(model.word_index("златом").unwrap_or_default() > 0, "check model working"),
            Err(err) => assert!(false, "failed to create a fasttext model {:?}", err),
        };
    }
}