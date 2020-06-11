'''
Preprocess manually tagged data
'''

import json

from preprocess_simple import build_encodings, build_dataset


def load_examples(data_path, dataset_name):
    examples_path = data_path + f"/{dataset_name}_examples.json"

    # load examples
    with open(examples_path, "r") as f:
        examples = json.load(f)

    return examples


def preprocess(data_path, examples_for_vocab=None):
    train_examples = load_examples(data_path, "manually_tagged_train")
    dev_examples = load_examples(data_path, "manually_tagged_dev")

    if examples_for_vocab:
        word_encoder, tag_encoder = build_encodings(examples_for_vocab)
    else:
        word_encoder, tag_encoder = build_encodings(train_examples)

    train_dataset = build_dataset(train_examples, word_encoder, tag_encoder)
    dev_dataset = build_dataset(dev_examples, word_encoder, tag_encoder)

    return train_dataset, dev_dataset, None, word_encoder, tag_encoder
