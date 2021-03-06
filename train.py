import os
from datetime import datetime
import subprocess
import json

import numpy as np

import tensorflow as tf
from tensorflow import keras

from preprocess_simple import preprocess as preprocess_simple
from preprocess_original import preprocess as preprocess_original
from preprocess_manual import preprocess as preprocess_manual
from preprocess_combined import preprocess as preprocess_combined
from build_model import build_model
from masked_accuracy import SparseCategoricalAccuracyMaskZeros
from evaluate import evaluate

# specify what data to use, and
# preprocess the data
# (tokenize and convert each token to an integer)
preprocessors = {
    'original':
    preprocess_original,  # nyt dataset with original tokenization method
    'simple':
    preprocess_simple,  # nyt dataset with a simplier tokenization method
    'manual':
    preprocess_manual,  # manually tagged dataset with same tokenization as 'simple
    'combined':
    preprocess_combined,  # combined dataset (nyt and manually tagged)
}

PREPROCESSOR = 'manual'
preprocess = preprocessors[PREPROCESSOR]

MANUAL_EXAMPLE_WEIGHT = 1

# how many training examples to train on - 169207 is the full nyt dataset
TRAIN_SIZE = 169207
# shuffle buffer - needs to be larger than TRAIN_SIZE for a perfect shuffle
SHUFFLE_BUFFER_SIZE = 200000

# load data and encoders, separate data into batches and pad
if PREPROCESSOR == 'combined':
    train_data, dev_data, _, word_encoder, tag_encoder = preprocess(
        "./data", MANUAL_EXAMPLE_WEIGHT)
    train_batches = (
        train_data.take(TRAIN_SIZE).shuffle(SHUFFLE_BUFFER_SIZE).padded_batch(
            128, padded_shapes=([None], [None], [None])))

else:
    train_data, dev_data, _, word_encoder, tag_encoder = preprocess("./data")
    train_batches = (
        train_data.take(TRAIN_SIZE).shuffle(SHUFFLE_BUFFER_SIZE).padded_batch(
            128, padded_shapes=([None], [None])))

dev_batches = dev_data.padded_batch(128, padded_shapes=([None], [None]))

# build model:
ARCHITECTURE = "lstm"
EMBEDDING_UNITS = 128
RECURRENT_UNITS = 512
NUM_RECURRENT_LAYERS = 2
REGULARIZER = None
REGULARIZATION_FACTOR = 0
DROPOUT_RATE = 0
RECURRENT_DROPOUT_RATE = 0

model = build_model(
    architecture=ARCHITECTURE,
    num_recurrent_layers=NUM_RECURRENT_LAYERS,
    embedding_units=EMBEDDING_UNITS,
    recurrent_units=RECURRENT_UNITS,
    regularizer=REGULARIZER,
    regularization_factor=REGULARIZATION_FACTOR,
    dropout_rate=DROPOUT_RATE,
    recurrent_dropout_rate=RECURRENT_DROPOUT_RATE,
    vocab_size=word_encoder.vocab_size,
    tag_size=tag_encoder.vocab_size,
)

# compile model:
OPTIMIZER = "adam"
model.compile(optimizer=OPTIMIZER,
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[SparseCategoricalAccuracyMaskZeros(name="accuracy")])

# make experiment directory and save experiment params down
date_string = datetime.now().strftime("%Y%m%d_%H%M")
commit_string = subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
experiment_dir = "experiments/" + date_string + "_" + commit_string
os.mkdir(experiment_dir)

# add tensorboard logs
epoch_tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=experiment_dir + "/epoch_logs", histogram_freq=1)

# fit model:
EPOCHS = 10
history = model.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=dev_batches,
                    validation_steps=40,
                    callbacks=[epoch_tensorboard_callback])

# save params down
with open(experiment_dir + "/params.json", "w") as f:
    json.dump(
        {
            "PREPROCESSOR": PREPROCESSOR,
            "MANUAL_EXAMPLE_WEIGHT": MANUAL_EXAMPLE_WEIGHT,
            "ARCHITECTURE": ARCHITECTURE,
            "EMBEDDING_UNITS": EMBEDDING_UNITS,
            "RECURRENT_UNITS": RECURRENT_UNITS,
            "NUM_RECURRENT_LAYERS": NUM_RECURRENT_LAYERS,
            "REGULARIZER": REGULARIZER,
            "REGULARIZATION_FACTOR": REGULARIZATION_FACTOR,
            "DROPOUT_RATE": DROPOUT_RATE,
            "RECURRENT_DROPOUT_RATE": RECURRENT_DROPOUT_RATE,
            "OPTIMIZER": OPTIMIZER,
            "EPOCHS": EPOCHS,
            "TRAIN_SIZE": TRAIN_SIZE,
            "SHUFFLE_BUFFER_SIZE": SHUFFLE_BUFFER_SIZE,
        },
        f,
        indent=4)

# save model weights for later usage
model.save_weights(experiment_dir + "/model_weights")

# evaluate model and save metrics:
evaluation = evaluate(model, train_batches, dev_batches,
                      PREPROCESSOR == 'combined')

with open(experiment_dir + "/results.json", "w") as f:
    json.dump(evaluation, f, indent=4)
