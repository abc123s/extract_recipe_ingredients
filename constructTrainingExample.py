import json

import tensorflow as tf
from tensorflow import keras

import pandas as pd

import numpy as np

from tokenizer import IngredientPhraseTokenizer, clean
from preprocess_simple import preprocess

from build_model import build_model
from masked_accuracy import SparseCategoricalAccuracyMaskZeros

ingredientPhraseTokenizer = IngredientPhraseTokenizer()

_, _, _, word_encoder, tag_encoder = preprocess("./data")

experiment_dir = "experiments/20200607_0646_53eb83b"

# load experiment params
with open(experiment_dir + "/params.json", "r") as f:
    params = json.load(f)

# build and compile model based on experiment params:
model = build_model(
    architecture=params["ARCHITECTURE"],
    embedding_units=params["EMBEDDING_UNITS"],
    num_recurrent_layers=params.get("NUM_RECURRENT_LAYERS", 1),
    recurrent_units=params["RECURRENT_UNITS"],
    regularizer=params.get("REGULARIZER", None),
    regularization_factor=params.get("REGULARIZATION_FACTOR", 0),
    dropout_rate=params.get("DROPOUT_RATE", 0),
    recurrent_dropout_rate=params["RECURRENT_DROPOUT_RATE"],
    vocab_size=word_encoder.vocab_size,
    tag_size=tag_encoder.vocab_size,
)

OPTIMIZER = params["OPTIMIZER"]
model.compile(optimizer=params["OPTIMIZER"],
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[SparseCategoricalAccuracyMaskZeros(name="accuracy")])

# load final weights from experiment into model:
model.load_weights(experiment_dir + "/model_weights")

# prepare examples
EXAMPLES_FILE = 'trainingExamples.csv'
examples = pd.read_csv(EXAMPLES_FILE)
examples = examples.fillna("")

clean_examples = []
tokens = []
encodedTokens = []
for _, example in examples.iterrows():
    ingredientPhrase = example["ingredientPhrase"]
    if ingredientPhrase != "":
        clean_examples.append(example)

        exampleTokens = ingredientPhraseTokenizer.tokenize(
            example["ingredientPhrase"])
        tokens.append(exampleTokens)

        exampleEncodedTokens = word_encoder.encode(example["ingredientPhrase"])
        encodedTokens.append(exampleEncodedTokens)


# make initial prediction for examples
def example_generator():
    for example in encodedTokens:
        yield example


example_batches = tf.data.Dataset.from_generator(
    example_generator,
    output_types=tf.int32).padded_batch(128, padded_shapes=[None])

encodedGuesses = []
for example_batch in example_batches:
    for model_output, example in zip(model.predict(example_batch),
                                     example_batch.numpy().tolist()):
        prediction = keras.backend.flatten(
            keras.backend.argmax(model_output)).numpy().tolist()
        padding_start = example.index(0) if example[-1] == 0 else len(example)
        encodedGuesses.append(prediction[0:padding_start])

guesses = []
for encodedGuess in encodedGuesses:
    guess = tag_encoder.decode(encodedGuess).split(' ')
    guesses.append(guess)

if len(tokens) != len(clean_examples) or len(tokens) != len(
        encodedTokens) or len(tokens) != len(guesses):
    print(
        'Uh oh, tokens, encodedTokens or guesses don\'t have the same length as the clean example list'
    )
    print(len(examples))
    print(len(tokens))
    print(len(encodedTokens))
    print(len(guesses))

trainingExamples = []
for index in range(len(clean_examples)):
    example = clean_examples[index]
    exampleTokens = tokens[index]
    exampleEncodedTokens = tokens[index]
    exampleGuess = guesses[index]

    if len(exampleTokens) != len(exampleEncodedTokens) or len(
            exampleTokens) != len(exampleGuess):
        print(
            'Uh oh, example has different number of tokens, encoded tokens, or guessed tags:'
        )
        print(example["ingredientPhrase"])
        print(exampleTokens)
        print(exampleEncodedTokens)
        print(exampleGuess)

    trainingExamples.append({
        "source": example["source"],
        "original": example["ingredientPhrase"],
        "tokens": exampleTokens,
        "guess": exampleGuess,
    })

with open("trainingExamples.json", "w") as f:
    json.dump(trainingExamples, f, indent=4)
