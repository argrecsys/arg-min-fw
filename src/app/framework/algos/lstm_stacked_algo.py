import pandas as pd
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Embedding, LSTM
from tf_keras.preprocessing import text


# Embedding + Stacked LSTM + Dense(sigmoid/softmax)
class StackedLSTMModel:

    def __init__(self, num_task: int):
        self.name = "Embedding + Stacked LSTM"
        self.num_task = num_task
        self.input_size = 1000

        if num_task == 1:
            self.last_activation = "sigmoid"
            self.last_units = 1
            self.num_labels = 2

        elif num_task == 2:
            self.last_activation = "softmax"
            self.last_units = 3
            self.num_labels = 3

        else:
            self.last_activation = None
            self.last_units = None

        self.hyperparams = {
            "learning_rate": {"min_value": 1e-7, "max_value": 1e-2, "dist": "log"},
            "epsilon": {"min_value": 1e-9, "max_value": 1e-6, "dist": "log"},
            "epochs": {"min_value": 1, "max_value": 50, "dist": "lineal", "step": 1},
            "batch_size": {
                "min_value": 16,
                "n": 4,
                "dist": "cat",
            },
            "num_layers": {"min_value": 1, "max_value": 5, "dist": "lineal", "step": 1},
            "num_units": {
                "min_value": 64,
                "max_value": 512,
                "dist": "lineal",
                "step": 64,
            },
            "num_embedding_units": {
                "min_value": 64,
                "max_value": 512,
                "dist": "lineal",
                "step": 64,
            },
        }

    def load_datasets(
        self,
        training_texts,
        training_labels,
        validation_texts,
        validation_labels,
        test_texts,
        test_labels,
        vocabulary_size: int = 1000,
    ):
        # Construction of an index (vocabulary) for the 1000 most frequent words in the training data set
        tokenizer = text.Tokenizer(num_words=vocabulary_size)
        tokenizer.fit_on_texts(training_texts)

        # Vectorization of texts using one-hot encoding representation
        x_training = tokenizer.texts_to_matrix(training_texts, mode="binary")
        y_training = training_labels
        x_validation = tokenizer.texts_to_matrix(validation_texts, mode="binary")
        y_validation = validation_labels
        x_test = tokenizer.texts_to_matrix(test_texts, mode="binary")
        y_test = test_labels

        self.input_size = x_training[0].shape[0]

        y_training = pd.to_numeric(y_training, errors="coerce")
        y_validation = pd.to_numeric(y_validation, errors="coerce")
        y_test = pd.to_numeric(y_test, errors="coerce")

        # Build final datasets
        self.training_ds = (x_training, y_training)
        self.validation_ds = (x_validation, y_validation)
        self.test_ds = (x_test, y_test)

    def get_ds_training(self) -> tuple:
        return self.training_ds

    def get_ds_validation(self) -> tuple:
        return self.validation_ds

    def get_ds_test(self) -> tuple:
        return self.test_ds

    def create_model(
        self,
        input_lstm_units: int,
        input_embedding_units: int,
        num_layers: int,
        vocabulary_size: int = 1000,
    ):
        model = Sequential()

        model.add(Embedding(vocabulary_size, input_embedding_units, mask_zero=True))
        for layer in range(num_layers - 1):
            model.add(LSTM(input_lstm_units, return_sequences=True))
        model.add(LSTM(input_lstm_units))
        model.add(Dense(units=self.last_units, activation=self.last_activation))

        return model

    def get_hyperparams(self):
        return self.hyperparams
