
import json
import csv
import random

import tensorflow as tf
from tensorflow import keras

from preprocess_simple import preprocess as preprocess_simple, load_examples as load_simple_examples
from preprocess_original import preprocess as preprocess_original, crfFile2Examples
from preprocess_manual import preprocess as preprocess_manual, load_examples as load_manual_examples
from preprocess_combined import preprocess as preprocess_combined
from build_model import build_model
from masked_accuracy import SparseCategoricalAccuracyMaskZeros

# modify this line to change which experiment directory you wish to 
# run an error analysis on
experiment_dir = "experiments/20200610_1547_6184893"

preprocessors = {
    'simple': preprocess_simple,
    'original': preprocess_original,
    'manual': preprocess_manual,
    'combined': preprocess_combined,
}

# used when running error analysis on a fine-tuned model
def load_original():
    return crfFile2Examples("./data/train.crf")


def load_simple():
    return load_simple_examples("./data", "train")


def load_combined():
    nyt_train_examples = load_simple_examples("./data", "train")

    manual_train_examples = load_manual_examples("./data",
                                                 "manually_tagged_train")
    return [
        *nyt_train_examples,
        *manual_train_examples,
    ]


def load_manual():
    return load_manual_examples("./data", "train")


example_loader = {
    'original': load_original,
    'simple': load_simple,
    'manual': load_manual,
    'combined': load_combined,
}

# load experiment params
with open(experiment_dir + "/params.json", "r") as f:
    params = json.load(f)

# grab dev set to do error analysis on, making sure to properly handle
# fine-tuned models (which use two different datasets)
if params.get("ORIGINAL_EXPERIMENT_DIR", None):
    original_examples = example_loader[params.get("PREPROCESSOR", "original")]()
    _, dev_data, _, word_encoder, tag_encoder = preprocess_manual(
        "./data", original_examples)
else:
    preprocess = preprocessors[params.get("PREPROCESSOR", 'original')]
    _, dev_data, _, word_encoder, tag_encoder = preprocess("./data")

# build and compile model based on experiment params:
model = build_model(
    architecture = params["ARCHITECTURE"],
    embedding_units = params["EMBEDDING_UNITS"],
    num_recurrent_layers = params.get("NUM_RECURRENT_LAYERS", 1),
    recurrent_units = params["RECURRENT_UNITS"],
    regularizer = params.get("REGULARIZER", None),
    regularization_factor = params.get("REGULARIZATION_FACTOR", 0),
    dropout_rate = params.get("DROPOUT_RATE", 0),
    recurrent_dropout_rate = params["RECURRENT_DROPOUT_RATE"],
    vocab_size = word_encoder.vocab_size,
    tag_size = tag_encoder.vocab_size,
)

OPTIMIZER = params["OPTIMIZER"]
model.compile(
    optimizer = params["OPTIMIZER"],
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = [SparseCategoricalAccuracyMaskZeros(name = "accuracy")]
)

# load final weights from experiment into model:
model.load_weights(experiment_dir + "/model_weights")

dev_batches = dev_data.padded_batch(128, padded_shapes = ([None], [None]))

# grab sample errors and compute confusion matrix
incorrect_sentences = []
all_predictions = []
all_labels = []

for sentences, labels in dev_batches:
    model_outputs = model.predict(sentences)
    for model_output, label, sentence in zip(model_outputs, labels, sentences):
        prediction = keras.backend.flatten(
            keras.backend.argmax(
                model_output
            )
        ).numpy().tolist()
        answer = label.numpy().tolist()

        # trim answer and prediction
        padding_start = answer.index(0) if answer[-1] == 0 else len(answer)
        trimmed_prediction = prediction[0:padding_start]
        trimmed_answer = answer[0:padding_start]

        # compute whether sentence is correct
        correct = [pred == label for pred, label in zip(trimmed_prediction, trimmed_answer)]

        # store labels and predictions in array for confusion matrix computation:
        all_predictions.extend(trimmed_prediction)
        all_labels.extend(trimmed_answer)

        # decode sentence and labels, and check if something is wrong with padding
        decoded_sentence = word_encoder.decode(sentence).split(' ')
        decoded_prediction = tag_encoder.decode(trimmed_prediction).split(' ')
        decoded_answer = tag_encoder.decode(trimmed_answer).split(' ')

        if len(answer) > len(decoded_sentence) and answer[len(decoded_sentence)] != 0:
            print('Uh oh - sentence has different number of elements than associated labels.')
            print(decoded_sentence)
            print(decoded_prediction)
            print(decoded_answer)

        if len(correct) != sum(correct):
            incorrect_sentences.append((
                decoded_sentence,
                decoded_prediction,
                decoded_answer,
            ))

# randomly sample sentence errors and write to csv for error analysis
sample_errors = random.choices(incorrect_sentences, k=100)

with open(experiment_dir + '/error_analysis.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    
    for sentence, prediction, answer in sample_errors:
        writer.writerow(sentence)
        writer.writerow(prediction)
        writer.writerow(answer)
        writer.writerow([])
        writer.writerow([])

# compute confusion matrix and write to csv
confusion_matrix = tf.math.confusion_matrix(all_labels, all_predictions).numpy().tolist()

with open(experiment_dir + '/confusion_matrix.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)

    writer.writerow([None, "PAD", *tag_encoder.decode(range(1, tag_encoder.vocab_size)).split(' ')])
    for index, confusion_matrix_row in enumerate(confusion_matrix):
        writer.writerow(["PAD" if index == 0 else tag_encoder.decode([index]), *confusion_matrix_row])




