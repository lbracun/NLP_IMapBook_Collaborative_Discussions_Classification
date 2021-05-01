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


def main():
    data_df, target_df = utils.load_discussions_data(keep_punctuation=True)

    # glove = KeyedVectors.load_word2vec_format(GLOVE_PATH, binary=False, no_header=True)

    majority_model = DummyClassifier(strategy="most_frequent")
    tfidf_model = TFIDFModel(regularization=0.01)
    # glove_model = GloveModel(glove, embedding_dim=100, n_filters=256, epochs=40)
    # bert_model = BertModel()

    models = [majority_model, tfidf_model]
    evaluations = utils.evaluate_models(models, data_df, target_df)
    print(json.dumps(evaluations, indent=2))


if __name__ == "__main__":
    main()


# {
#     "DummyClassifier(strategy='most_frequent')": {
#         "fit_time": 0.0021196603775024414,
#         "score_time": 0.006248891353607178,
#         "test_accuracy": 0.46848905438952837,
#         "test_f1_macro": 0.04253689415292923,
#         "test_neg_log_loss": -18.357737701887633,
#     },
#     "TFIDFModel(regularization=0.01)": {
#         "fit_time": 4.65557986497879,
#         "score_time": 0.17913484573364258,
#         "test_accuracy": 0.665865493116678,
#         "test_f1_macro": 0.4190023077913235,
#         "test_neg_log_loss": -1.2442675188634662,
#     },
#     "GloveModel(embedding_dim=100, epochs=40, glove=<gensim.models.keyedvectors.KeyedVectors object at 0x7f53625bd940>)": {
#         "fit_time": 81.70379132032394,
#         "score_time": 0.6193323135375977,
#         "test_accuracy": 0.6313924621981494,
#         "test_f1_macro": 0.3428168250525275,
#         "test_neg_log_loss": -8.493166933771246,
#     },
# }
