import tensorflow as tf
from transformers import InputFeatures
from transformers import TFBertForSequenceClassification, BertTokenizer


# Relu 16 + Sigmoid 1
class BETOClassifierModel:

    def __init__(self, num_task: int):
        self.name = "BETO-based Pretraned model"
        self.num_task = num_task
        self.model_name = "dccuchile/bert-base-spanish-wwm-cased"

        if num_task == 1:
            self.num_labels = 2

        elif num_task == 2:
            self.num_labels = 3

        else:
            self.last_activation = None
            self.last_units = None

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

        self.hyperparams = {
            "learning_rate": {"min_value": 1e-7, "max_value": 1e-2, "dist": "log"},
            "epsilon": {"min_value": 1e-9, "max_value": 1e-6, "dist": "log"},
            "epochs": {"min_value": 1, "max_value": 50, "dist": "lineal", "step": 1},
            "batch_size": {
                "min_value": 16,
                "n": 4,
                "dist": "cat",
            },
        }

    from transformers import InputFeatures

    def __convert_examples_to_features(self, texts, labels):
        labels = list(labels)
        batch_encoding = self.tokenizer.batch_encode_plus(
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

    def load_datasets(
        self,
        training_texts,
        training_labels,
        validation_texts,
        validation_labels,
        test_texts,
        test_labels,
        batch_size=32,
    ):
        train_features = self.__convert_examples_to_features(
            training_texts, training_labels
        )
        train_dataset = self.__convert_features_to_tf_dataset(train_features)

        validation_features = self.__convert_examples_to_features(
            validation_texts, validation_labels
        )
        validation_dataset = self.__convert_features_to_tf_dataset(validation_features)

        test_features = self.__convert_examples_to_features(test_texts, test_labels)
        test_dataset = self.__convert_features_to_tf_dataset(test_features)

        train_dataset = train_dataset.shuffle(100).batch(batch_size)
        validation_dataset = validation_dataset.batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)

        # Build final datasets
        self.training_ds = train_dataset
        self.validation_ds = validation_dataset
        self.test_ds = test_dataset

    def get_ds_training(self) -> tf.data.Dataset:
        return self.training_ds

    def get_ds_validation(self) -> tf.data.Dataset:
        return self.validation_ds

    def get_ds_test(self) -> tf.data.Dataset:
        return self.test_ds

    def create_model(self):
        model = TFBertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
        return model

    def get_hyperparams(self):
        return self.hyperparams
