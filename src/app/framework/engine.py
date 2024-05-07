# Basic Python libraries
import time
from datetime import datetime
import pandas as pd
from functools import partial

# ML/DL libraries
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tf_keras
from tf_keras.callbacks import EarlyStopping
import optuna

# Modules
from .algos.ffn_dense_algo import FFNDenseModel
from .algos.ffn_dense2_algo import FFNDense2Model


class Engine:

    def __init__(self, task_number: int, verbose: bool = False) -> None:
        self.task_number = task_number
        self.verbose = verbose
        self.sent_column = "sentence"
        self.target_column = "label"
        self.model_list = [FFNDenseModel, FFNDense2Model]

    def __print(self, message) -> None:
        if self.verbose:
            current_time_utc = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{current_time_utc }: {message}")

    def __get_hyperparams(self, trial, hyperparam_setup) -> dict:
        hyperparams = {}

        # Learning rate
        hp_name = "learning_rate"
        learning_rate_setup = hyperparam_setup.get(hp_name)
        hp_learning_rate = trial.suggest_float(
            hp_name,
            learning_rate_setup["min_value"],
            learning_rate_setup["max_value"],
            log=(learning_rate_setup["dist"] == "log"),
        )
        hyperparams[hp_name] = hp_learning_rate

        # Epsilon
        hp_name = "epsilon"
        epsilon_setup = hyperparam_setup.get(hp_name)
        hp_epsilon = trial.suggest_float(
            hp_name,
            epsilon_setup["min_value"],
            epsilon_setup["max_value"],
            log=(learning_rate_setup["dist"] == "log"),
        )
        hyperparams[hp_name] = hp_epsilon

        # Number of epochs
        hp_name = "epochs"
        epochs_setup = hyperparam_setup.get(hp_name)
        hp_epochs = trial.suggest_int(
            hp_name,
            epochs_setup["min_value"],
            epochs_setup["max_value"],
            step=epochs_setup["step"],
        )
        hyperparams[hp_name] = hp_epochs

        # Batch size
        hp_name = "batch_size"
        batch_size_setup = hyperparam_setup.get(hp_name)
        sequence = [
            batch_size_setup["min_value"] * (2**i) for i in range(batch_size_setup["n"])
        ]
        hp_batch_size = trial.suggest_categorical(hp_name, sequence)
        hyperparams[hp_name] = hp_batch_size

        # Number of hidden layers
        hp_name = "num_layers"
        num_layers_setup = hyperparam_setup.get(hp_name)
        hp_num_layers = trial.suggest_int(
            hp_name,
            num_layers_setup["min_value"],
            num_layers_setup["max_value"],
            step=num_layers_setup["step"],
        )
        hyperparams[hp_name] = hp_num_layers

        # Number of units per layer
        hp_name = "num_units"
        num_units_setup = hyperparam_setup.get(hp_name)
        hp_num_units = trial.suggest_int(
            hp_name,
            num_units_setup["min_value"],
            num_units_setup["max_value"],
            step=num_units_setup["step"],
        )
        hyperparams[hp_name] = hp_num_units

        # Dropout rate
        hp_name = "dropout"
        dropout_setup = hyperparam_setup.get(hp_name)
        if dropout_setup:
            hp_dropout = trial.suggest_float(
                hp_name,
                dropout_setup["min_value"],
                dropout_setup["max_value"],
                log=(dropout_setup["dist"] == "log"),
                step=dropout_setup["step"],
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
            monitor="val_accuracy", patience=5, mode="max", verbose=1
        )

        # Create ML/DL model
        if hp_dropout is not None:
            model = algorithm.create_model(hp_num_units, hp_num_layers, hp_dropout)
        else:
            model = algorithm.create_model(hp_num_units, hp_num_layers)
        model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        )

        # Train and evaluate using tf.keras.Model.fit()
        history = model.fit(
            x=train_dataset[0],
            y=train_dataset[1],
            batch_size=hp_batch_size,
            epochs=hp_epochs,
            validation_data=validation_dataset,
            callbacks=[early_stopping],
            verbose=False,
        )

        return -history.history["val_accuracy"][-1]

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

    def run_optimization(self, n_jobs: int = 1, n_trials: int = 10) -> None:
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
