import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import utils
from bert_model import BertModel
from gensim.models import KeyedVectors
from glove_model import GloveModel
from sklearn.dummy import DummyClassifier
from tfidf_model import TFIDFModel

GLOVE_PATH = "../models/glove.6B.100d.txt"

CUSTOM_NAMES = [
    utils.COL_BOOK_SIMILARITY,
    utils.COL_CONTAINS_EMOTICON,
    utils.COL_CONTAINS_LINK,
    utils.COL_WORD_COUNT,
    utils.COL_CHAR_COUNT,
    utils.COL_UPPERCASE_COUNT,
    # utils.COL_QUESTION_COUNT,  # Intentional as it slightly hurts the performance
]


# TF-IDF / Logistic regression model evaluation should be fast.


def main():
    data_df, target_df = utils.load_discussions_data(keep_punctuation=True)

    # glove = KeyedVectors.load_word2vec_format(GLOVE_PATH, binary=False, no_header=True)

    majority_model = DummyClassifier(strategy="most_frequent")
    tfidf_model = TFIDFModel(regularization=0.1, custom_feature_names=CUSTOM_NAMES, max_iter=2500, solver="sag")
    # glove_model = GloveModel(glove, embedding_dim=100, n_filters=256, epochs=40)
    # bert_model = BertModel()

    models = [
        majority_model,
        tfidf_model,
        # glove_model,
        # bret_model,
    ]
    evaluations = utils.evaluate_models(models, data_df, target_df)
    print(json.dumps(evaluations, indent=2))


if __name__ == "__main__":
    main()
