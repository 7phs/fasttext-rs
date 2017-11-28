use fasttext::*;
use predict::PredictRecord;
use std::path::Path;

static UNKNOWN_PATH: &'static str = "unknown path";
static UNSUPERVISED_MODEL_PATH: &'static str = "./test-data/unsupervised_model.bin";
static UNSUPERVISED_VECTORS_PATH: &'static str = "./test-data/unsupervised_model.vec";
static SUPERVISED_MODEL_PATH: &'static str = "./test-data/supervised_model.bin";
static SUPERVISED_VECTORS_PATH: &'static str = "./test-data/supervised_model.vec";

fn path(p: &str) -> &Path {
    Path::new(p)
}

#[test]
fn test_fasttext_default() {
    FastText::default();
}

#[test]
fn test_fasttext_load_model() {
    let mut model = FastText::default();

    match model.load_model(path(UNKNOWN_PATH)) {
        Ok(_) => assert!(false, "failed to raise an error for an unknown model path"),
        Err(_) => assert!(true),
    }

    match model.load_model(path(UNSUPERVISED_MODEL_PATH)) {
        Ok(_) => assert!(true),
        Err(err) => assert!(false, err),
    }
}

#[test]
fn test_fasttext_load_vectors() {
    let mut model = FastText::default();

    match model.load_model(path(UNSUPERVISED_MODEL_PATH)) {
        Ok(_) => assert!(true),
        Err(err) => {
            println!("Failed to load model {:?} with error {:?}", UNSUPERVISED_MODEL_PATH, err);
            assert!(false)
        }
    }

    match model.load_vectors(path(UNKNOWN_PATH)) {
        Ok(_) => {
            println!("failed to raise an error for an unknown vectors path");
            assert!(false)
        }
        Err(_) => assert!(true),
    }

    match model.load_vectors(path(UNSUPERVISED_VECTORS_PATH)) {
        Ok(_) => assert!(true),
        Err(err) => {
            println!("Failed to load vectors {:?} with error {:?}", UNSUPERVISED_VECTORS_PATH, err);
            assert!(false)
        }
    }
}

#[test]
fn test_fasttext_get_dictionary() {
    let mut model = FastText::default();

    match model.load_model(path(UNSUPERVISED_MODEL_PATH)) {
        Ok(_) => assert!(true),
        Err(err) => {
            println!("Failed to load model {:?} with error {:?}", UNSUPERVISED_MODEL_PATH, err);
            assert!(false)
        }
    }

    match model.load_vectors(path(UNSUPERVISED_VECTORS_PATH)) {
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

    let expected_word = String::from("златом");
    let expected_index: i64 = 22;

    match dict.word_index(expected_word.as_str()) {
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
    let mut model = FastText::default();

    match model.load_model(path(UNSUPERVISED_MODEL_PATH)) {
        Ok(_) => assert!(true),
        Err(err) => {
            println!("Failed to load model {:?} with error {:?}", UNSUPERVISED_MODEL_PATH, err);
            assert!(false)
        }
    }

    match model.load_vectors(path(UNSUPERVISED_VECTORS_PATH)) {
        Ok(_) => assert!(true),
        Err(err) => {
            println!("Failed to load vectors {:?} with error {:?}", UNSUPERVISED_VECTORS_PATH, err);
            assert!(false)
        }
    }

    let word = String::from("златом");
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
    const COEFF: f32 = 1000.0;

    match model.word_to_vector(word.as_str()) {
        Some(vec) => assert_eq!(
            vec.as_slice().iter()
                .map(|v| (v * COEFF).trunc())
                .collect::<Vec<f32>>(),
            expected_vec.iter()
                .map(|v| (v * COEFF).trunc())
                .collect::<Vec<f32>>()),
        None => {
            println!("failed to get vector for word {:?}", word);
            assert!(false)
        }
    };
}

#[test]
fn test_fasttext_get_sentence_vector() {
    let mut model = FastText::default();

    match model.load_model(path(UNSUPERVISED_MODEL_PATH)) {
        Ok(_) => assert!(true),
        Err(err) => {
            println!("Failed to load model {:?} with error {:?}", UNSUPERVISED_MODEL_PATH, err);
            assert!(false)
        }
    }

    match model.load_vectors(path(UNSUPERVISED_VECTORS_PATH)) {
        Ok(_) => assert!(true),
        Err(err) => {
            println!("Failed to load vectors {:?} with error {:?}", UNSUPERVISED_VECTORS_PATH, err);
            assert!(false)
        }
    }

    let text = String::from("Кащей над златом чахнет");
    let expected_vec = vec![
        0.037130516f32, -0.023536012, -0.05462671, -0.0033913944, -0.018964574, 0.06834176, 0.11407065, 0.16753219, 0.054430895, 0.08919889,
        0.062001314, 0.12970737, -0.08993994, -0.0013820566, 0.0024787933, 0.015299875, 0.043955993, -0.02965922, -0.002137018, -0.033652093,
        0.03392063, -0.084329106, 0.07451887, 0.020768367, -0.020052189, 0.009265052, -0.020497253, 0.13506308, 0.027629945, -0.036651403,
        -0.053276077, 0.00411354, 0.035692926, 0.025573859, 0.015886994, 0.046156306, 0.03847931, -0.013042267, -0.016854543, 0.017981935,
        0.03060818, 0.009657327, -0.13549992, -0.02106245, 0.018689439, 0.0015525157, 0.0126991235, 0.03377308, 0.045630272, 0.047137648,
        0.014620029, 0.1009623, -0.05738701, -0.053834576, 0.06026704, -0.011569845, 0.02681889, -0.027683325, -0.0058202855, -0.010907434,
        -0.060490265, -0.0024461383, 0.10436508, -0.0076349233, 0.019921903, -0.038126707, 0.03616117, 0.0049869185, 0.027135931, 0.026441898,
        0.027044544, 0.013437899, 0.019479826, 0.051475823, 0.06197407, -0.045852486, -0.016118113, 0.10718948, 0.017485319, -0.02544314,
        -0.026579026, 0.11026137, 0.064151205, 0.0035600364, -0.0493346, 0.021935625, -0.010017559, 0.04056901, 0.04047651, -0.0039925557,
        0.029625436, 0.030468367, -0.07004885, -0.028973147, 0.0024994463, -0.032818604, -0.092352696, -0.014490175, 0.07920408, -0.0061761905
    ];
    const COEFF: f32 = 1000.0;

    match model.sentence_to_vector(text.as_str()) {
        Some(vec) => assert_eq!(
            vec.as_slice().iter()
                .map(|v| (v * COEFF).trunc())
                .collect::<Vec<f32>>(),
            expected_vec.as_slice().iter()
                .map(|v| (v * COEFF).trunc())
                .collect::<Vec<f32>>(),
            "check word vector"
        ),
        None => assert!(false, "failed to get vector for sentence {:?}", text),
    };
}

#[test]
fn test_fasttext_predict_unsupervised() {
    let mut model = FastText::default();

    match model.load_model(path(UNSUPERVISED_MODEL_PATH)) {
        Ok(_) => assert!(true),
        Err(err) => assert!(false, "Failed to load model {:?} with error {:?}", UNSUPERVISED_MODEL_PATH, err),
    }

    match model.load_vectors(path(UNSUPERVISED_VECTORS_PATH)) {
        Ok(_) => assert!(true),
        Err(err) => assert!(false, "Failed to load vectors {:?} with error {:?}", UNSUPERVISED_VECTORS_PATH, err),
    }

    let text = String::from("Куда сходить вечером");

    match model.predict(text.as_str(), 10) {
        Ok(_) => assert!(false, "predict wasn't implemented unimplgot empty result for sentence {:?}", text),
        Err(_) => assert!(true),
    };
}

#[test]
fn test_fasttext_predict_supervised() {
    let mut model = FastText::default();

    match model.load_model(path(SUPERVISED_MODEL_PATH)) {
        Ok(_) => assert!(true),
        Err(err) => assert!(false, "Failed to load model {:?} with error {:?}", SUPERVISED_MODEL_PATH, err),
    }

    match model.load_vectors(path(SUPERVISED_VECTORS_PATH)) {
        Ok(_) => assert!(true),
        Err(err) => assert!(false, "Failed to load vectors {:?} with error {:?}", SUPERVISED_VECTORS_PATH, err),
    }

    let text = String::from("Куда сходить вечером");
    let expected_result = vec![
        PredictRecord::new(-1.1025261, "__label__пожелание"),
        PredictRecord::new(-1.1025261, "__label__вопрос"),
        PredictRecord::new(-1.1025261, "__label__приветствие"),
    ];

    match model.predict(text.as_str(), 10) {
        Ok(result) => {
            if result.is_empty() {
                assert!(false, "got empty result for sentence {:?}", text);
            } else {
                assert_eq!(
                    result.as_slice(),
                    expected_result.as_slice(),
                    "check predict result"
                );
            }
        },
        Err(err) => assert!(false, "failed to predict for sentence {:?}, {:?}", text, err),
    };
}