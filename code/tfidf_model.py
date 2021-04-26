from nltk.corpus import stopwords
from sklearn import pipeline
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class TFIDFModel(BaseEstimator):
    def __init__(self, max_features=None, min_df=1, max_df=1.0, regularization=1.0, max_iter=500):
        # We have to set the parameters in __init__ for cloning.
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.regularization = regularization
        self.max_iter = max_iter
        self._model: pipeline.Pipeline = pipeline.make_pipeline(
            TfidfVectorizer(
                token_pattern="[^\s]+",
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df,
                stop_words=stopwords.words("english"),
            ),
            LogisticRegression(
                multi_class="multinomial",
                C=1 / self.regularization,
                max_iter=self.max_iter,
            ),
        )

    def fit(self, messages, target):
        self._model.fit(messages, target)
        return self

    def predict(self, messages):
        return self._model.predict(messages)

    def predict_proba(self, messages):
        return self._model.predict_proba(messages)
