from tf_keras.models import Sequential
from tf_keras.layers import Dense
from tf_keras.preprocessing import text


# Relu 16 + Sigmoid 1
class NN1Model:

    def __init__(self, num_task: int):
        self.name = "nn1_algo"
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

        self.hyperparam = {
            "learning_rate": {"min_value": 1e-6, "max_value": 1e-4, "dist": "log"},
            "epsilon": {"min_value": 1e-9, "max_value": 1e-7, "dist": "log"},
            "epochs": {"min_value": 10, "max_value": 100, "dist": "lineal", "step": 1},
            "num_layers": {"min_value": 1, "max_value": 5, "dist": "lineal", "step": 1},
            "units": {"min_value": 2, "max_value": 16, "dist": "lineal", "step": 1},
            "batch_size": {
                "min_value": 2,
                "max_value": 16,
                "dist": "lineal",
                "step": 1,
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
        self.x_training = tokenizer.texts_to_matrix(training_texts, mode="binary")
        self.y_training = training_labels
        self.x_validation = tokenizer.texts_to_matrix(validation_texts, mode="binary")
        self.y_validation = validation_labels
        self.x_test = tokenizer.texts_to_matrix(test_texts, mode="binary")
        self.y_test = test_labels

        self.input_size = self.x_training[0].shape[0]

    def create_model(self):
        model = Sequential()
        model.add(Dense(units=16, activation="relu", input_shape=(self.input_size,)))
        model.add(Dense(units=self.last_units, activation=self.last_activation))
        self.model = model

    def get_model(self):
        return self.model

    def get_hyper_params(self):
        return self.hyperparam

    def get_model_output_size(self):
        return self.last_units
