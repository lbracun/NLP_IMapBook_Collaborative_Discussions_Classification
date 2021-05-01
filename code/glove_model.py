import numpy as np
import tensorflow as tf
import utils
from sklearn.base import BaseEstimator
from tensorflow.keras import layers, regularizers
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


class GloveModel(BaseEstimator):
    def __init__(
        self, glove, embedding_dim, epochs=20, n_filters=256, kernel_size=3, pool_size=3, dense_units=128, dropout=0.4
    ):
        self.glove = glove
        self.embedding_dim = embedding_dim
        self.n_filters = n_filters
        self.epochs = epochs
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dense_units = dense_units
        self.dropout = dropout

        self._vectorizer = TextVectorization(max_tokens=1625, output_sequence_length=250)
        self._target_to_index = {}
        self._index_to_target = {}

    def fit(self, data_df, target_df, verbose=0):
        messages_ds = tf.data.Dataset.from_tensor_slices(data_df[utils.COL_MESSAGE]).batch(128)
        self._vectorizer.adapt(messages_ds)
        vocab = self._vectorizer.get_vocabulary()
        word_index = dict(zip(vocab, range(len(vocab))))

        n_tokens = len(vocab) + 2
        misses = []
        embedding_matrix = np.zeros((n_tokens, self.embedding_dim))
        for word, index in word_index.items():
            if word in self.glove:
                embedding_matrix[index] = self.glove[word]
            else:
                misses.append(word)

        self._index_to_target = dict(enumerate(set(target_df)))
        self._target_to_index = {v: k for k, v in self._index_to_target.items()}
        target_idx = target_df.map(self._target_to_index)

        x_train = self._transform(data_df)
        y_train = np.array(target_idx)

        self._model = self._make_model(embedding_matrix, len(set(target_df)))
        self._model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"])
        self._model.fit(x_train, y_train, epochs=self.epochs, batch_size=128, verbose=verbose)
        return self

    def _transform(self, data_df):
        x_msgs = self._vectorizer(np.array([[s] for s in data_df[utils.COL_MESSAGE]])).numpy()
        return x_msgs

    def predict(self, data_df):
        x_features = self._transform(data_df)
        y_idx = self._model.predict(x_features)
        return np.array([self._index_to_target[np.argmax(y)] for y in y_idx])

    def predict_proba(self, data_df):
        x_features = self._transform(data_df)
        return self._model.predict(x_features)

    def _make_model(self, embedding_matrix, n_classes):
        n_tokens, embedding_dim = embedding_matrix.shape
        model = tf.keras.models.Sequential()
        model.add(
            Embedding(n_tokens, embedding_dim, embeddings_initializer=Constant(embedding_matrix), trainable=False)
        )
        model.add(
            layers.Conv1D(self.n_filters, self.kernel_size, activation="relu", kernel_regularizer=regularizers.l2(1e-8))
        )
        model.add(layers.MaxPooling1D(self.pool_size))
        model.add(
            layers.Conv1D(self.n_filters, self.kernel_size, activation="relu", kernel_regularizer=regularizers.l2(1e-8))
        )
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dense(self.dense_units, activation="relu", kernel_regularizer=regularizers.l2(1e-4)))
        model.add(layers.Dropout(self.dropout))
        model.add(layers.Dense(n_classes, activation="softmax", kernel_regularizer=regularizers.l2(1e-4)))
        return model
