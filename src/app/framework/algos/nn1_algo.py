import pandas as pd
from keras.api.models import Sequential
from keras.api.layers import Dense


# Relu 16 + Sigmoid 1
class NN1Model:

    def __init__(self, num_task: int):
        self.name = "nn1_algo"
        self.num_task = num_task
        self.input_size = 1

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

    def load_dataset(self):
        pass

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

    def model_compile(self):
        self.model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        self.model.summary()

    def model_fit(self, y_training, y_validation, y_test):
        y_training = pd.to_numeric(y_training, errors="coerce")
        y_validation = pd.to_numeric(y_validation, errors="coerce")
        y_test = pd.to_numeric(y_test, errors="coerce")
        # history = self.model.fit(x_training, y_training, epochs=100, batch_size=32, validation_data=(x_validation, y_validation), verbose=1)
        return None  # history
