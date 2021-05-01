import numpy as np
import utils
from nltk.corpus import stopwords
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class TFIDFModel(BaseEstimator):
    def __init__(self, max_features=None, min_df=1, max_df=1.0, regularization=1.0, max_iter=500, pca_components=3):
        # We have to set the parameters in __init__ for cloning.
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.regularization = regularization
        self.max_iter = max_iter
        self.pca_components = pca_components

        make_vectorizer = lambda: TfidfVectorizer(
            token_pattern="[^\s]+",
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words=stopwords.words("english"),
        )

        self._msg_vect = make_vectorizer()
        self._resp_vect = make_vectorizer()
        self._log_reg = LogisticRegression(
            multi_class="multinomial",
            C=1 / self.regularization,
            max_iter=self.max_iter,
        )

    def _transform(self, data_df):
        x_msg = self._msg_vect.transform(data_df[utils.COL_MESSAGE])
        x_resp = self._resp_vect.transform(data_df[utils.COL_COLLAB_RESP])
        return sparse.hstack((x_msg, x_resp))

    def fit(self, data_df, target_df):
        # joined_txt = data_df[utils.COL_MESSAGE].append(data_df[utils.COL_COLLAB_RESP])
        self._msg_vect.fit(data_df[utils.COL_MESSAGE])
        self._resp_vect.fit(data_df[utils.COL_COLLAB_RESP])
        x_features = self._transform(data_df)
        self._log_reg.fit(x_features, target_df)
        return self

    def predict(self, data_df):
        x_features = self._transform(data_df)
        return self._log_reg.predict(x_features)

    def predict_proba(self, data_df):
        x_features = self._transform(data_df)
        return self._log_reg.predict_proba(x_features)
