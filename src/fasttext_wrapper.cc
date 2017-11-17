#include <iostream>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <unistd.h>
#include "fasttext/src/fasttext.h"

const int RES_OK = 0;
const int RES_ERROR_NOT_OPEN = 1;
const int RES_ERROR_WRONG_MODEL = 2;
const int RES_ERROR_NOT_INIT = 3;

extern "C" {
    struct WrapperDictionary {
        const fasttext::Dictionary *dict;
    };

    struct WrapperFastText {
        fasttext::FastText *model;
    };

    struct WrapperVector {
        fasttext::Vector *vector;
    };

    struct WrapperString {
        char*        str;
        unsigned int len;
        unsigned int cap;
    };

    struct WrapperPredictRecord {
        float       predict;
        const char* word;
    };

    struct WrapperPredictResult {
        const WrapperPredictRecord* records;
        const char*                 err;

        std::vector<WrapperPredictRecord> records_;
        std::vector<std::string>          words_;
        std::string                       err_;
    };
}

bool checkModelInitialization(const struct WrapperFastText* wrapper) {
    if (wrapper==nullptr ||
        wrapper->model==nullptr ||
        wrapper->model->getDictionary()==nullptr
    ) {
        return false;
    }

    return true;
}

bool checkModelFile(std::istream& in) {
    int32_t magic, version;

    in.read((char*)&(magic), sizeof(int32_t));
    if (magic != FASTTEXT_FILEFORMAT_MAGIC_INT32) {
        return false;
    }

    in.read((char*)&(version), sizeof(int32_t));
    if (version > FASTTEXT_VERSION) {
        return false;
    }

    return true;
}

int checkVectorsFile(const std::string& path, const int ndim) {
    std::ifstream in(path);

    int64_t n, dim;
    if (!in.is_open()) {
        return RES_ERROR_NOT_OPEN;
    }

    in >> n >> dim;
    if (dim != ndim) {
        return RES_ERROR_WRONG_MODEL;
    }

    in.close();

    return RES_OK;
}

void stringInit(struct WrapperString *wrapper, const std::string& str) {
    strncpy(wrapper->str, str.c_str(), wrapper->cap);
    wrapper->len = str.length();
}

void predictResultResize(struct WrapperPredictResult* result, size_t sz) {
    result->records_.resize(sz);
    result->words_.resize(sz);
}

void predictResultSet(struct WrapperPredictResult* result, size_t i, std::pair<float, std::string>& rec) {
    auto& new_rec = result->records_[i];
    auto& new_word = result->words_[i];

    new_rec.predict = std::get<0>(rec);
    new_word = std::get<1>(rec);
}

void predictResultSetError(struct WrapperPredictResult* result, const char* str) {
    result->err_ = std::string(str);
    result->err = result->err_.c_str();
}

void predictResultFinish(struct WrapperPredictResult* result) {
    auto& records = result->records_;
    auto& words = result->words_;

    for(size_t i = 0, sz = records.size(); i < sz; i++) {
        records[i].word = words[i].c_str();
    }

    result->records = records.data();
}

struct WrapperVector* Vector(int ndim) {
    WrapperVector *wrapper = (WrapperVector *)malloc(sizeof (struct WrapperVector));

    wrapper->vector = new fasttext::Vector(ndim);

    return wrapper;
}

struct WrapperDictionary* Dictionary(std::shared_ptr<const fasttext::Dictionary> dict) {
    WrapperDictionary *wrapper = (WrapperDictionary *)malloc(sizeof (struct WrapperDictionary));

    wrapper->dict = dict.get();

    return wrapper;
}

extern "C" {
    int DICT_Find(const struct WrapperDictionary* wrapper, const char* word) {
        return int(wrapper->dict->getId(word));
    }

    void DICT_GetWord(const struct WrapperDictionary* wrapper, int id, struct WrapperString *str) {
        stringInit(str, wrapper->dict->getWord(id));
    }

    int DICT_WordsCount(const struct WrapperDictionary* wrapper) {
        return wrapper->dict->nwords();
    }

    void VEC_Release(struct WrapperVector* wrapper) {
        delete wrapper->vector;

        free(wrapper);
    }

    int VEC_Len(struct WrapperVector* wrapper) {
        return wrapper->vector->size();
    }

    const float* VEC_GetData(struct WrapperVector* wrapper) {
        return wrapper->vector->data_;
    }

    int PRDCT_Len(const struct WrapperPredictResult* wrapper) {
        return wrapper->records_.size();
    }

    const struct WrapperPredictRecord* PRDCT_Records(const struct WrapperPredictResult* wrapper) {
        return wrapper->records;
    }

    const char* PRDCT_Error(const struct WrapperPredictResult* wrapper) {
        return wrapper->err;
    }

    void PRDCT_Release(const struct WrapperPredictResult* result) {
        delete result;
    }

    struct WrapperFastText* NewFastText() {
        WrapperFastText *wrapper = (WrapperFastText *)malloc(sizeof (struct WrapperFastText));

        wrapper->model = new fasttext::FastText();

        return wrapper;
    }

    int FT_LoadModel(struct WrapperFastText* wrapper, const char* path) {
        std::ifstream ifs(path, std::ifstream::binary);

        if (!ifs.good()) {
            return RES_ERROR_NOT_OPEN;
        }

        if (!checkModelFile(ifs)) {
            return RES_ERROR_WRONG_MODEL;
        }

        wrapper->model->loadModel(ifs);

        ifs.close();

        return RES_OK;
    }

    int FT_LoadVectors(struct WrapperFastText* wrapper, const char* path) {
        std::string vectorsPath(path);

        if (!checkModelInitialization(wrapper)) {
            return RES_ERROR_NOT_INIT;
        }

        const int res = checkVectorsFile(vectorsPath, wrapper->model->getDimension());
        if (res!=RES_OK) {
            return res;
        }

        wrapper->model->loadVectors(vectorsPath);

        return RES_OK;
    }

    struct WrapperDictionary* FT_GetDictionary(const struct WrapperFastText* wrapper) {
        return Dictionary(wrapper->model->getDictionary());
    }

    struct WrapperVector* FT_GetWordVector(const struct WrapperFastText* wrapper, const char* word) {
        struct WrapperVector* wrap_vector = Vector(wrapper->model->getDimension());

        wrapper->model->getWordVector(*wrap_vector->vector, word);

        return wrap_vector;
    }

    struct WrapperVector* FT_GetSentenceVector(const struct WrapperFastText* wrapper, const char* text) {
        std::istringstream str(text);

        struct WrapperVector* wrap_vector = Vector(wrapper->model->getDimension());

        wrapper->model->getSentenceVector(str, *wrap_vector->vector);

        return wrap_vector;
    }

    const struct WrapperPredictResult* FT_Predict(struct WrapperFastText* wrapper, const char* text, int k) {
        std::istringstream str(text);
        std::vector<std::pair<float, std::string>> prediction;

        struct WrapperPredictResult* result = new struct WrapperPredictResult();

        try {
            wrapper->model->predict(str, k, prediction);

            predictResultResize(result, prediction.size());
            for(size_t i = 0, sz = prediction.size(); i<sz; i++) {
                predictResultSet(result, i, prediction[i]);
            }

            predictResultFinish(result);
        } catch(std::exception &e) {
            predictResultSetError(result, e.what());
        }

        return result;
    }

    void FT_Release(struct WrapperFastText* wrapper) {
        delete wrapper->model;

        free(wrapper);
    }
}

extern "C" {
    const struct WrapperVector* test_VEC_New(short empty, int sz) {
        auto wrapper = Vector(sz);

        if(empty) {
            wrapper->vector->zero();
        } else {
            for(unsigned int i=0; i<wrapper->vector->size(); i++) {
                wrapper->vector->data_[i] = 1. + (1./float(i + 0.0001)) * (float(i) * float(i) * 1.019238);
            }
        }

        return wrapper;
    }

    const struct WrapperPredictResult* test_PRDCT_New(short err, int sz) {
        srand(42);

        auto wrapper = new struct WrapperPredictResult();

        if(err) {
            predictResultSetError(wrapper, "test error");
        }

        auto words = std::vector<std::string> {
            "слово",
            "о",
            "полку",
            "Игореве",
            "Бояна",
            "сказ"
        };

        predictResultResize(wrapper, sz);

        for(unsigned int i=0; i<sz; i++) {
            int r = rand();
            auto rec = std::pair<float, std::string>(float(r)/float(RAND_MAX), words[r % words.size()]);

            predictResultSet(wrapper, i, rec);
        }

        predictResultFinish(wrapper);

        return wrapper;
    }
}
