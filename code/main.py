import pandas as pd
import utils
from sklearn.dummy import DummyClassifier
from tfidf_model import TFIDFModel
from bert_model import BertModel


def main():
    data_df = utils.load_annotated_discussions_data(keep_punctuation=True)

    messages = data_df[utils.COL_MESSAGE]
    target = data_df[utils.COL_TARGET]

    majority_model = DummyClassifier(strategy="most_frequent")
    tfidf_model = TFIDFModel(regularization=0.01)
    bert_model = BertModel()

    models = [majority_model, tfidf_model, bert_model]

    evaluations = utils.evaluate_models(models, messages, target)
    for model_name, eval in evaluations.items():
        print(model_name, eval)


if __name__ == "__main__":
    main()
