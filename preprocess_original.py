import os

import tensorflow as tf
import tensorflow_datasets as tfds

TokenTextEncoder = tfds.features.text.TokenTextEncoder
Tokenizer = tfds.features.text.Tokenizer

# read examples from NYT dataset
def crfFile2Examples(file_name):
    examples = []

    with open(file_name, "r") as f:
        example = [[], []]

        for line in f:
            split_line = line.split()

            if len(split_line) > 0:
                word, *_, tag = split_line
                example[0].append(word)
                example[1].append(tag)

            elif len(example[0]) > 0:
                examples.append(example)
                
                example = [[], []]

        if len(example[0]) > 0:
            examples.append(example)

        f.close()
    
    return examples

# for now, just take the pre-tokenized words provided by the
# NYT dataset
class CustomTokenizer(Tokenizer):
    def tokenize(self, s):
        s = tf.compat.as_text(s)
        return [s]

def build_encodings(examples):
    vocab_list = sorted(set([word for example in examples for word in example[0]]))
    tag_list = sorted(set([tag for example in examples for tag in example[1]]))
    
    word_encoder = TokenTextEncoder(
        vocab_list, tokenizer = CustomTokenizer()
    )
    tag_encoder = TokenTextEncoder(
        tag_list, oov_buckets = 0, tokenizer = CustomTokenizer()
    )

    return word_encoder, tag_encoder

def build_dataset(examples, word_encoder, tag_encoder):
    def example_generator():
        for example in examples:
            # TODO: build a better encoder that doesn't require these
            # weird hacks - custom splitting, etc.
            yield (
                [word_encoder.encode(word)[0] if len(word_encoder.encode(word)) else word_encoder.encode("UNK")[0] for word in example[0]],
                [tag_encoder.encode(tag)[0] for tag in example[1]],
            )

    return tf.data.Dataset.from_generator(
        example_generator,
        output_types=(tf.int32, tf.int32)
    )

def preprocess(data_path):
    train_examples = crfFile2Examples(data_path + "/train.crf")
    dev_examples = crfFile2Examples(data_path + "/dev.crf")
    test_examples = crfFile2Examples(data_path + "/test.crf")

    word_encoder, tag_encoder = build_encodings(train_examples)

    '''
    encoding_path = data_path + "/encodings"
    word_encoder_path = encoding_path + "/word_encoder"
    tag_encoder_path = encoding_path + "/tag_encoder"
    if (os.path.exists(encoding_path)):
        word_encoder = TokenTextEncoder.load_from_file(word_encoder_path)
        tag_encoder = TokenTextEncoder.load_from_file(tag_encoder_path)
    else:
        os.mkdir(encoding_path)
        word_encoder, tag_encoder = build_encodings(train_examples)
        word_encoder.save_to_file(word_encoder_path)
        tag_encoder.save_to_file(tag_encoder_path)
    '''

    train_dataset = build_dataset(train_examples, word_encoder, tag_encoder)
    dev_dataset = build_dataset(dev_examples, word_encoder, tag_encoder)
    test_dataset = build_dataset(test_examples, word_encoder, tag_encoder)

    return train_dataset, dev_dataset, test_dataset, word_encoder, tag_encoder
