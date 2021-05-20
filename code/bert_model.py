import tensorflow as tf
import numpy as np

from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel, TFBertPreTrainedModel, TFBertMainLayer
from tensorflow.keras import layers



class BertModel(BaseEstimator):
    def __init__(self, output_classes=15, dropout_rate=0.2, cnn_filters=100, dnn_units=256, epochs=5):
        # We have to set the parameters in __init__ for cloning.
        self.epochs = epochs
        self.cnn_filters = cnn_filters
        self.dropout_rate = dropout_rate
        self.output_classes = output_classes
        self.dnn_units = dnn_units
         
        # set tokenizer
        self._tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # set model
        self._model = BertCustomEmbeddingModel.from_pretrained('bert-base-uncased', 
                                cnn_filters=self.cnn_filters,
                                dnn_units=self.dnn_units,
                                model_output_classes=self.output_classes,
                                dropout_rate=self.dropout_rate)

        if output_classes == 2:
            self._model.compile(loss="binary_crossentropy",
                                optimizer="adam",
                                metrics=["accuracy"])
        else:
            self._model.compile(loss="sparse_categorical_crossentropy",
                                optimizer="adam",
                                metrics=["sparse_categorical_accuracy"])

        

        # set LabelEncoder for encoding/decoding target values string<->number
        self.label_encoder = preprocessing.LabelEncoder()


        

    def fit(self, messages, target):
        # encode targets, so they can be decoded later
        target = self.label_encoder.fit_transform(target)

        # tokenize data
        train_token_ids = self.get_token_ids(messages)

        # create tf dataset
        train_data = tf.data.Dataset.from_tensor_slices((tf.constant(train_token_ids), tf.constant(target))).batch(12)

        # train model
        self._model.fit(train_data, epochs=self.epochs)

        return self

    def predict(self, messages):
        # extract values from probabilities
        result = np.argmax(self.predict_proba(messages), axis=1)

        # decode classes from numerical values
        return self.label_encoder.inverse_transform(result)

    def predict_proba(self, messages):
        # tokenize data
        test_token_ids = self.get_token_ids(messages)

        # create tf dataset
        test_data = tf.data.Dataset.from_tensor_slices((tf.constant(test_token_ids), tf.constant(np.ones(len(messages))))).batch(12)

        # run model prediction
        return self._model.predict(test_data)

    # from labs 09 - Transformers and BERT
    def get_token_ids(self, texts):
        return self._tokenizer.batch_encode_plus(texts['Message'], 
                                            add_special_tokens=True, 
                                            max_length = 128, 
                                            pad_to_max_length = True)["input_ids"]


# from labs 09 - Transformers and BERT
class BertCustomEmbeddingModel(TFBertPreTrainedModel):
    def __init__(self, config,
                 cnn_filters=50,
                 dnn_units=512,
                 model_output_classes=2,
                 dropout_rate=0.1,
                 training=False,
                 name="text_model",
                 *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = TFBertMainLayer(config, name="bert", trainable = False)
        
        self.cnn_layer1 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=2,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer2 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=3,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer3 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=4,
                                        padding="valid",
                                        activation="relu")
        self.pool = layers.GlobalMaxPool1D()
        
        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if model_output_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=model_output_classes,
                                           activation="softmax")

    def call(self, inputs, training = False, **kwargs):        
        bert_outputs = self.bert(inputs, training = training, **kwargs)
        
        l_1 = self.cnn_layer1(bert_outputs[0]) 
        l_1 = self.pool(l_1) 
        l_2 = self.cnn_layer2(bert_outputs[0]) 
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(bert_outputs[0])
        l_3 = self.pool(l_3) 
        
        concatenated = tf.concat([l_1, l_2, l_3], axis=-1) # (batch_size, 3 * cnn_filters)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)
        
        return model_output