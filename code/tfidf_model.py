import numpy as np
import utils
from nltk.corpus import stopwords
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class TFIDFModel(BaseEstimator):
    def __init__(
        self,
        max_features=None,
        min_df=1,
        max_df=1.0,
        regularization=1.0,
        max_iter=500,
        custom_feature_names=None,
        solver="lbfgs",
    ):
        # We need to save the parameters in __init__ for cloning.
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.regularization = regularization
        self.max_iter = max_iter
        self.solver = solver
        self.custom_feature_names = custom_feature_names

        make_vectorizer = lambda: TfidfVectorizer(
            token_pattern="[^\s]+",
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words=stopwords.words("english"),
        )

        self._msg_vect = make_vectorizer()
        self._resp_vect = make_vectorizer()
        self._scaler = StandardScaler(with_mean=False)
        self._log_reg = LogisticRegression(
            multi_class="multinomial",
            C=1 / self.regularization,
            max_iter=self.max_iter,
            solver=self.solver,
        )

    def _transform(self, data_df, fit=False):
        if fit:
            self._msg_vect.fit(data_df[utils.COL_MESSAGE])
            self._resp_vect.fit(data_df[utils.COL_COLLAB_RESP])

        x_msg = self._msg_vect.transform(data_df[utils.COL_MESSAGE])
        x_resp = self._resp_vect.transform(data_df[utils.COL_COLLAB_RESP])

        matrices = (x_msg, x_resp)

        if self.custom_feature_names:
            x_custom = sparse.csr_matrix(data_df[self.custom_feature_names])
            if fit:
                self._scaler.fit(x_custom)
            x_custom = self._scaler.transform(x_custom)

            matrices = (x_custom, x_msg, x_resp)

        x_features = sparse.hstack(matrices)
        return x_features

    def fit(self, data_df, target_df):
        x_features = self._transform(data_df, fit=True)
        self._log_reg.fit(x_features, target_df)
        return self

    def predict(self, data_df):
        x_features = self._transform(data_df)
        return self._log_reg.predict(x_features)

    def predict_proba(self, data_df):
        x_features = self._transform(data_df)
        return self._log_reg.predict_proba(x_features)
