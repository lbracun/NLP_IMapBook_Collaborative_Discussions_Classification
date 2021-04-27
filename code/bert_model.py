import tensorflow as tf
import numpy as np

from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from transformers import TFBertForSequenceClassification, BertTokenizer



class BertModel(BaseEstimator):
    def __init__(self, train=False):
        # We have to set the parameters in __init__ for cloning.
        self.train = train

        # set model
        self._model = TFBertForSequenceClassification.from_pretrained('../models/bert/', num_labels=15)
        
        # set tokenizer
        self._tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

        # compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        self._model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

        # set LabelEncoder for encoding/decoding target values string<->number
        self.label_encoder = preprocessing.LabelEncoder()


        

    def fit(self, messages, target):
        # encode targets, so they can be decoded later
        target = self.label_encoder.fit_transform(target)
    
        # no need to train, because we pretrained it (model in /models/bert)
        if self.train:
            # split to train and validation data
            X_train, X_val, y_train, y_val = train_test_split(messages, target, test_size=0.2, random_state=13)

            # tokenize data
            X_train_input = self.convert_to_input(X_train)
            X_val_input = self.convert_to_input(X_val)

            # create tf.dataset
            train_ds = tf.data.Dataset.from_tensor_slices((X_train_input[0],X_train_input[1],X_train_input[2],y_train)).map(self.example_to_features).shuffle(100).batch(12).repeat(5)
            val_ds=tf.data.Dataset.from_tensor_slices((X_val_input[0],X_val_input[1],X_val_input[2],y_val)).map(self.example_to_features).batch(12)

            self._model.fit(train_ds, epochs=3, validation_data=val_ds)

        return self

    def predict(self, messages):
        # extract values from probabilities
        result = np.argmax(self.predict_proba(messages), axis=1)

        # decode classes from numerical values
        return self.label_encoder.inverse_transform(result)

    def predict_proba(self, messages):
        # tokenize data
        X_input = self.convert_to_input(messages)
        # create tf.dataset
        X_ds = tf.data.Dataset.from_tensor_slices((X_input[0],X_input[1],X_input[2], np.ones(len(messages)))).map(self.example_to_features).batch(12)

        return self._model.predict(X_ds).logits


    # from labs 09 - Transformers and BERT
    def convert_to_input(self, messages, pad_token=0, pad_token_segment_id=0, max_length=128):
        input_ids, attention_masks,token_type_ids=[],[],[]

        for message in messages:
            inputs = self._tokenizer.encode_plus(message, add_special_tokens=True, max_length=max_length)

            i, t = inputs["input_ids"], inputs["token_type_ids"]
            m = [1] * len(i)

            padding_length = max_length - len(i)

            i = i + ([pad_token] * padding_length)
            m = m + ([0] * padding_length)
            t = t + ([pad_token_segment_id] * padding_length)
            
            input_ids.append(i)
            attention_masks.append(m)
            token_type_ids.append(t)

        return [np.asarray(input_ids), 
                np.asarray(attention_masks), 
                np.asarray(token_type_ids)]


    # from labs 09 - Transformers and BERT
    def example_to_features(self, input_ids, attention_masks, token_type_ids, y):
        return {"input_ids": input_ids,
                "attention_mask": attention_masks,
                "token_type_ids": token_type_ids},y