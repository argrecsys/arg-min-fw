from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import optuna
import tf_keras
from functools import partial

from .algos.nn1_algo import NN1Model


class Engine:

    def __init__(self, task_number: int, verbose: bool = False) -> None:
        self.task_number = task_number
        self.verbose = verbose
        self.sent_column = "sentence"
        self.target_column = "label"
        self.model_list = [NN1Model]

    def __print(self, values) -> None:
        if self.verbose:
            print(values)

    def __objective(self, trial, algorithm):

        # Model variables
        model = algorithm.create_model()
        train_dataset = algorithm.get_ds_training()
        validation_dataset = algorithm.get_ds_validation()

        # Adjustable hyperparameters
        hp_learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
        hp_epsilon = trial.suggest_float("epsilon", 1e-9, 1e-7, log=True)
        hp_epochs = trial.suggest_int("epochs", 2, 7, step=1)
        optimizer = tf_keras.optimizers.Adam(
            learning_rate=hp_learning_rate, epsilon=hp_epsilon, clipnorm=1.0
        )

        # Fixed hyperparameters
        loss = tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf_keras.metrics.SparseCategoricalAccuracy("accuracy")

        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

        # Train and evaluate using tf.keras.Model.fit()
        history = model.fit(
            train_dataset, validation_data=validation_dataset, epochs=hp_epochs
        )

        return history.history["val_accuracy"][-1]

    def load_data(
        self, dataset_path: str, text_column: str, label_column: str, label_dict: dict
    ) -> None:
        self.dataset = pd.read_csv(dataset_path)
        self.dataset = self.dataset[[text_column, label_column]]
        self.dataset = self.dataset.rename(
            columns={text_column: self.sent_column, label_column: self.target_column}
        )
        self.dataset = self.dataset.replace({self.target_column: label_dict})
        self.__print(self.dataset)

    def split_data(
        self,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        shuffled: bool = False,
        random_state: int = None,
    ) -> None:
        df = self.dataset

        # Split the data into a temporary train set and test/validation set
        train, temp = train_test_split(
            df,
            test_size=(test_size + validation_size),
            stratify=df[self.target_column],
            random_state=random_state,
        )

        # Split the temp set into validation and test sets
        test, validation = train_test_split(
            temp,
            test_size=validation_size / (test_size + validation_size),
            stratify=temp[self.target_column],
            random_state=random_state,
        )

        # Shuffle datasets
        if shuffled:
            training_set = shuffle(train)
            validation_set = shuffle(validation)
            test_set = shuffle(test)
        else:
            training_set = train
            validation_set = validation
            test_set = test

        self.__print(f"Training size: {len(training_set)}")
        self.__print(f"Validation size: {len(validation_set)}")
        self.__print(f"Test size: {len(test_set)}")

        self.train_texts = training_set[self.sent_column]
        self.train_labels = training_set[self.target_column]

        self.validation_texts = validation_set[self.sent_column]
        self.validation_labels = validation_set[self.target_column]

        self.test_texts = test_set[self.sent_column]
        self.test_labels = test_set[self.target_column]

    def load_algorithms(self) -> int:
        self.algorithms = {}

        for model_type in self.model_list:
            algo = model_type(self.task_number)
            algo.load_datasets(
                self.train_texts,
                self.train_labels,
                self.validation_texts,
                self.validation_labels,
                self.test_texts,
                self.test_labels,
            )
            self.algorithms[algo.name] = algo

        return len(self.algorithms)

    def run_optimization(self, n_trials: int = 5) -> None:

        for name, algorithm in self.algorithms.items():
            print(f"{datetime.now()} - Optimizing model: {name}")
            objective_with_model = partial(self.__objective, algorithm=algorithm)

            study = optuna.create_study(direction="minimize")
            study.optimize(objective_with_model, n_trials)
