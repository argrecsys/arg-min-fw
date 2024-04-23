import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from transformers import InputFeatures

# from algos.nn1_algo import MM1Model


class Engine:

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.sent_column = "sentence"
        self.target_column = "label"

    def __print(self, values) -> None:
        if self.verbose:
            print(values)

    def __convert_examples_to_features(self, tokenizer, texts, labels):
        labels = list(labels)
        batch_encoding = tokenizer.batch_encode_plus(
            texts, max_length=128, padding="longest"
        )

        features = []
        for i in range(len(texts)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}
            feature = InputFeatures(**inputs, label=labels[i])
            features.append(feature)

        return features

    def __convert_features_to_tf_dataset(self, features):
        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    ex.label,
                )

        output_types = (
            {
                "input_ids": tf.int32,
                "attention_mask": tf.int32,
                "token_type_ids": tf.int32,
            },
            tf.int32,
        )

        output_shapes = (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        )

        dataset = tf.data.Dataset.from_generator(gen, output_types, output_shapes)
        return dataset

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
            self.training_set = shuffle(train)
            self.validation_set = shuffle(validation)
            self.test_set = shuffle(test)
        else:
            self.training_set = train
            self.validation_set = validation
            self.test_set = test

        self.__print(f"Training size: {len(self.training_set)}")
        self.__print(f"Validation size: {len(self.validation_set)}")
        self.__print(f"Test size: {len(self.test_set)}")

    def create_features(self) -> None:
        train_texts = self.training_set[self.sent_column]
        train_labels = self.training_set[self.target_column]

        validation_texts = self.validation_set[self.sent_column]
        validation_labels = self.validation_set[self.target_column]

        test_texts = self.test_set[self.sent_column]
        test_labels = self.test_set[self.target_column]
