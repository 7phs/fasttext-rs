#[cfg(test)]
use fasttext::*;
use std::path::Path;
use std::ops::Deref;

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

#[test]
fn test_fasttext_get_word_vector() {
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

    let word = "златом";
    let expected_vec = vec![
        -0.00093162473f32, -0.00064210495, -0.0009707032, 0.00031974318, -0.0002315091, 0.002165826, 0.0018783867, 0.0021941606, 0.0014265366, -0.0004747169,
        0.00042756647, 0.002485942, -3.925598e-05, 0.0019237917, -0.0005152924, 0.00038866568, 0.0009962652, 1.5210654e-05, -0.0002595325, -0.00020531945,
        0.00032480282, -0.0004953152, 0.0011179452, 0.00047672325, -0.001567682, 0.0011290247, -0.001348938, 0.0013085243, 0.00067846774, -0.0006734705,
        -5.052518e-05, -0.0015279177, -0.00070518075, 0.00029840754, 0.00085332844, 0.00039165834, 0.0004000477, 0.00023921908, 0.0010118099, -0.00033629616,
        0.00089929136, 0.0023271097, -0.0025798841, 0.0011896471, 0.0010260209, -0.00046896562, -5.8911148e-05, 0.0013999777, 0.002503657, -0.00042554422,
        0.0006372212, 0.0020875847, -0.0013018411, 0.00032486717, 0.0005404552, 0.000112594964, 0.0009362643, -0.0021627878, 0.0002152612, -0.00052340183,
        -0.00093787076, -0.0017059175, 0.00025377644, 0.00017843576, -0.0014451658, -0.0015405576, 0.00087846775, 0.0005363762, 0.00039133723, 0.002590361,
        0.00010595202, 0.001551432, 0.0008010692, -0.00014902855, -0.0018039232, -0.0013080842, -0.00027898262, 0.0011848757, 0.0003816138, 0.00019682113,
        0.002425637, 0.0033935579, 0.00022636236, 0.00014700758, -0.00064028014, -0.00031854503, -0.00012771421, 0.00016985959, 0.0013882271, -0.0021626558,
        0.0010302362, 0.0005801475, -0.0024974225, -0.00091070565, -0.00015146619, -0.0018224111, -0.0011335025, -9.8327204e-05, 0.00093681796, 0.00021071793,
    ];
    const coeff: f32 = 1000.0;

    match model.word_to_vector(word) {
        Some(vec) => assert_eq!(
            vec.deref().iter()
                .map(|v| (v * coeff).trunc())
                .collect::<Vec<f32>>(),
            expected_vec.iter()
                .map(|v| (v * coeff).trunc())
                .collect::<Vec<f32>>()),
        None => {
            println!("failed to get vector for word {:?}", word);
            assert!(false)
        }
    }
}