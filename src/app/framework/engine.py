import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class Engine:

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def __print(self, values):
        if self.verbose:
            print(values)

    def load_data(
        self, dataset_path: str, text_column: str, label_column: str, label_dict: dict
    ):
        self.dataset = pd.read_csv(dataset_path)
        self.dataset = self.dataset[[text_column, label_column]]
        self.dataset = self.dataset.rename(
            columns={text_column: "sentence", label_column: "label"}
        )
        self.dataset = self.dataset.replace({"label": label_dict})
        self.__print(self.dataset)

    def split_data(
        self,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        shuffled: bool = False,
        random_state: int = None,
    ):
        df = self.dataset
        target_column = "label"

        # Split the data into a temporary train set and test/validation set
        train, temp = train_test_split(
            df,
            test_size=(test_size + validation_size),
            stratify=df[target_column],
            random_state=random_state,
        )

        # Split the temp set into validation and test sets
        test, validation = train_test_split(
            temp,
            test_size=validation_size / (test_size + validation_size),
            stratify=temp[target_column],
            random_state=random_state,
        )

        if shuffled:
            self.training_set = shuffle(train)
            self.validation_set = shuffle(validation)
            self.test_set = shuffle(test)
        else:
            self.training_set = train
            self.validation_set = validation
            self.test_set = test
