'''
Preprocess NYT dataset using the "simple" preprocessor that uses a cleaner tokenization method, 
and manually tagged examples, and combine the two datasets, giving a higher weight to the manually
tagged examples
'''
import tensorflow as tf

from preprocess_manual import load_examples as load_manual_examples, build_encodings, build_dataset
from preprocess_simple import load_examples as load_nyt_examples


def build_weighted_dataset(examples_and_weights, word_encoder, tag_encoder):
    weighted_examples = []
    for examples, weight in examples_and_weights:
        for example in examples:
            weighted_examples.append((example[0], example[1], [weight]))

    def example_generator():
        for example in weighted_examples:
            # TODO: build a better encoder that doesn't require these
            # weird hacks - custom splitting, etc.
            yield (word_encoder.encode(example[0]),
                   [tag_encoder.encode(tag)[0]
                    for tag in example[1]], example[2])

    return tf.data.Dataset.from_generator(example_generator,
                                          output_types=(tf.int32, tf.int32,
                                                        tf.int32))


def preprocess(data_path, manual_weight=5):
    nyt_train_examples = load_nyt_examples(data_path, "train")

    manual_train_examples = load_manual_examples(data_path,
                                                 "manually_tagged_train")
    manual_dev_examples = load_manual_examples(data_path,
                                               "manually_tagged_dev")

    word_encoder, tag_encoder = build_encodings([
        *nyt_train_examples,
        *manual_train_examples,
    ])

    train_dataset = build_weighted_dataset([
        (nyt_train_examples, 1),
        (manual_train_examples, manual_weight),
    ], word_encoder, tag_encoder)
    dev_dataset = build_dataset(manual_dev_examples, word_encoder, tag_encoder)

    return train_dataset, dev_dataset, None, word_encoder, tag_encoder
