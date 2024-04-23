import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class Engine:

    def __init__(self):
        pass

    def load_dataset(self, dataset_path: str):
        self.dataset = pd.read_csv(dataset_path)
        print(self.dataset)

    def split_data(
        df, target_column, test_size=0.2, validation_size=0.1, random_state=None
    ):

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

        return train, validation, test
