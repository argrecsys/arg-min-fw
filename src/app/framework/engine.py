# Basic Python libraries
import time
from datetime import datetime
import pandas as pd
from functools import partial
import logging

# ML/DL libraries
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tf_keras
from tf_keras.callbacks import EarlyStopping
import optuna

# Modules
from .algos.ffn_dense_algo import FFNDenseModel
from .algos.ffn_dense2_algo import FFNDense2Model
from .algos.lstm_stacked_algo import StackedLSTMModel
from .algos.lstm_bidirectional_algo import BiLSTMModel
from .algos.beto_classifier_algo import BETOClassifierModel

logging.basicConfig(
    level=logging.INFO,
    filename="src/log/engine_log.log",
    filemode="a",
    format="%(asctime)s %(levelname)s %(message)s",
)


class Engine:

    def __init__(self, task_number: int, verbose: bool = False) -> None:
        self.task_number = task_number
        self.verbose = verbose
        self.sent_column = "sentence"
        self.target_column = "label"
        self.model_list = [
            FFNDenseModel,
            FFNDense2Model,
            StackedLSTMModel,
            BiLSTMModel,
            BETOClassifierModel,
        ]
        self.__print(">> Starting ArgMining framework")

    def __print(self, message) -> None:
        if self.verbose:
            logging.info(message)

    def __get_hyperparams(self, trial, hyperparam_setup) -> dict:
        hyperparams = {}

        # Learning rate
        hp_name = "learning_rate"
        setup_learning_rate = hyperparam_setup.get(hp_name)
        hp_learning_rate = trial.suggest_float(
            hp_name,
            setup_learning_rate["min_value"],
            setup_learning_rate["max_value"],
            log=(setup_learning_rate["dist"] == "log"),
        )
        hyperparams[hp_name] = hp_learning_rate

        # Epsilon
        hp_name = "epsilon"
        setup_epsilon = hyperparam_setup.get(hp_name)
        hp_epsilon = trial.suggest_float(
            hp_name,
            setup_epsilon["min_value"],
            setup_epsilon["max_value"],
            log=(setup_epsilon["dist"] == "log"),
        )
        hyperparams[hp_name] = hp_epsilon

        # Number of epochs
        hp_name = "epochs"
        setup_epochs = hyperparam_setup.get(hp_name)
        hp_epochs = trial.suggest_int(
            hp_name,
            setup_epochs["min_value"],
            setup_epochs["max_value"],
            step=setup_epochs["step"],
        )
        hyperparams[hp_name] = hp_epochs

        # Batch size
        hp_name = "batch_size"
        setup_batch_size = hyperparam_setup.get(hp_name)
        sequence = [
            setup_batch_size["min_value"] * (2**i) for i in range(setup_batch_size["n"])
        ]
        hp_batch_size = trial.suggest_categorical(hp_name, sequence)
        hyperparams[hp_name] = hp_batch_size

        # Number of units per layer (optional)
        hp_name = "num_units"
        setup_num_units = hyperparam_setup.get(hp_name)
        if setup_num_units:
            hp_num_units = trial.suggest_int(
                hp_name,
                setup_num_units["min_value"],
                setup_num_units["max_value"],
                step=setup_num_units["step"],
            )
            hyperparams[hp_name] = hp_num_units
        else:
            hyperparams[hp_name] = None

        # Number of hidden layers (optional)
        hp_name = "num_layers"
        setup_num_layers = hyperparam_setup.get(hp_name)
        if setup_num_layers:
            hp_num_layers = trial.suggest_int(
                hp_name,
                setup_num_layers["min_value"],
                setup_num_layers["max_value"],
                step=setup_num_layers["step"],
            )
            hyperparams[hp_name] = hp_num_layers
        else:
            hyperparams[hp_name] = None

        # Number of embedding units (optional)
        hp_name = "num_embedding_units"
        setup_num_embedding_units = hyperparam_setup.get(hp_name)
        if setup_num_embedding_units:
            hp_num_embedding_units = trial.suggest_int(
                hp_name,
                setup_num_embedding_units["min_value"],
                setup_num_embedding_units["max_value"],
                step=setup_num_embedding_units["step"],
            )
            hyperparams[hp_name] = hp_num_embedding_units
        else:
            hyperparams[hp_name] = None

        # Dropout rate (optional)
        hp_name = "dropout"
        setup_dropout = hyperparam_setup.get(hp_name)
        if setup_dropout:
            hp_dropout = trial.suggest_float(
                hp_name,
                setup_dropout["min_value"],
                setup_dropout["max_value"],
                log=(setup_dropout["dist"] == "log"),
                step=setup_dropout["step"],
            )
            hyperparams[hp_name] = hp_dropout
        else:
            hyperparams[hp_name] = None

        return hyperparams

    def __objective(self, trial, algorithm):

        # Datasets
        train_dataset = algorithm.get_ds_training()
        validation_dataset = algorithm.get_ds_validation()

        # Adjustable hyperparameters
        hyperparams = self.__get_hyperparams(trial, algorithm.get_hyperparams())
        hp_learning_rate = hyperparams["learning_rate"]
        hp_epsilon = hyperparams["epsilon"]
        hp_epochs = hyperparams["epochs"]
        hp_batch_size = hyperparams["batch_size"]
        hp_num_units = hyperparams["num_units"]
        hp_num_layers = hyperparams["num_layers"]
        hp_num_embedding_units = hyperparams["num_embedding_units"]
        hp_dropout = hyperparams["dropout"]

        # Fixed hyperparameters
        optimizer = tf_keras.optimizers.Adam(
            learning_rate=hp_learning_rate, epsilon=hp_epsilon, clipnorm=1.0
        )
        # loss = tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # metric = tf_keras.metrics.SparseCategoricalAccuracy("accuracy")
        # metrics = [tf_keras.metrics.BinaryAccuracy(), tf_keras.metrics.FalseNegatives()]

        # Define EarlyStopping callback
        early_stopping = EarlyStopping(
            monitor="val_accuracy", patience=5, mode="max", verbose=self.verbose
        )

        # Create ML/DL model
        if hp_num_units is None and hp_num_layers is None:
            model = algorithm.create_model()

        elif hp_num_embedding_units is not None:
            if hp_num_layers is None:
                model = algorithm.create_model(hp_num_units, hp_num_embedding_units)
            else:
                model = algorithm.create_model(
                    hp_num_units, hp_num_embedding_units, hp_num_layers
                )

        else:
            if hp_dropout is None:
                model = algorithm.create_model(hp_num_units, hp_num_layers)
            else:
                model = algorithm.create_model(hp_num_units, hp_num_layers, hp_dropout)

        model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        )

        # Train and evaluate using tf.keras.Model.fit()
        if hp_num_units is None and hp_num_layers is None:

            # Pretrained-based models
            history = model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=hp_epochs,
                batch_size=hp_batch_size,
                callbacks=[early_stopping],
                verbose=self.verbose,
            )

            metric_value = history.history["val_accuracy"][-1]

        else:
            # FFN- and LSTM-based models
            history = model.fit(
                x=train_dataset[0],
                y=train_dataset[1],
                validation_data=validation_dataset,
                epochs=hp_epochs,
                batch_size=hp_batch_size,
                callbacks=[early_stopping],
                verbose=self.verbose,
            )

            metric_value = -history.history["val_accuracy"][-1]

        return metric_value

    def load_data(
        self, dataset_path: str, text_column: str, label_column: str, label_dict: dict
    ) -> None:
        self.__print(">> Loading dataset")
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
        self.__print(">> Loading framework algorithms")
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

        n_algo = len(self.algorithms)
        self.__print(f">> {n_algo} has been loaded")
        return n_algo

    def run_optimization(self, n_jobs: int = 1, n_trials: int = 10) -> None:
        self.__print(">> Start framework optimization")
        start_time = time.time()

        for index, name in enumerate(self.algorithms):
            algorithm = self.algorithms[name]
            study_name = f"{name}-{index}"
            self.__print(f"{datetime.now()} - Optimizing model: {study_name}")

            # Optimaze model
            objective_with_model = partial(self.__objective, algorithm=algorithm)
            study = optuna.create_study(study_name=study_name, direction="minimize")
            study.optimize(objective_with_model, n_trials=n_trials, n_jobs=n_jobs)

            # Display best result
            best_trial = study.best_trial
            self.__print(f"Number of finished trials: {len(study.trials)}")
            self.__print("Best trial:")
            self.__print(f"  Value: {-best_trial.value}")
            self.__print("  Params:")
            for key, value in best_trial.params.items():
                self.__print(f"    {key}: {value}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        self.__print(f"Elapsed time: {elapsed_time} seconds")
        self.__print("End framework optimization")
        self.__print("")
