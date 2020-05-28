import os
from datetime import datetime
import subprocess
import json

import numpy as np

import tensorflow as tf
from tensorflow import keras

from preprocess import preprocess
from build_model import build_model
from masked_accuracy import SparseCategoricalAccuracyMaskZeros
from evaluate import evaluate

train_data, dev_data, _, word_encoder, tag_encoder = preprocess("./data")

TRAIN_SIZE = 169207
SHUFFLE_BUFFER_SIZE = 200000
train_batches = (train_data
    .take(TRAIN_SIZE)
    .shuffle(SHUFFLE_BUFFER_SIZE)
    .padded_batch(128, padded_shapes = ([None], [None]))
)

dev_batches = dev_data.padded_batch(128, padded_shapes = ([None], [None]))

# build model:
ARCHITECTURE = "rnn"
EMBEDDING_UNITS = 128
RECURRENT_UNITS = 512
DROPOUT_RATE = 0.2
RECURRENT_DROPOUT_RATE = 0.2

model = build_model(
    architecture = ARCHITECTURE,
    embedding_units = EMBEDDING_UNITS,
    recurrent_units = RECURRENT_UNITS,
    dropout_rate = DROPOUT_RATE,
    recurrent_dropout_rate = RECURRENT_DROPOUT_RATE,
    vocab_size = word_encoder.vocab_size,
    tag_size = tag_encoder.vocab_size,
)

# compile model:
OPTIMIZER = "adam"
model.compile(
    optimizer=OPTIMIZER,
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics=[SparseCategoricalAccuracyMaskZeros(name = "accuracy")]
)

# make experiment directory and save experiment params down
date_string = datetime.now().strftime("%Y%m%d_%H%M")
commit_string = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
experiment_dir = "experiments/" + date_string + "_" + commit_string
os.mkdir(experiment_dir)

# add tensorboard logs
epoch_tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = experiment_dir + "/epoch_logs", histogram_freq=1)

# fit model:
EPOCHS = 10
history = model.fit(
    train_batches,
    epochs = EPOCHS,
    validation_data=dev_batches,
    validation_steps=8,
    callbacks = [epoch_tensorboard_callback]
)

# save params down
with open(experiment_dir + "/params.json", "w") as f:
    json.dump(
        {
            "ARCHITECTURE": ARCHITECTURE,
            "EMBEDDING_UNITS": EMBEDDING_UNITS,
            "RECURRENT_UNITS": RECURRENT_UNITS,
            "DROPOUT_RATE": DROPOUT_RATE,
            "RECURRENT_DROPOUT_RATE": RECURRENT_DROPOUT_RATE,
            "OPTIMIZER": OPTIMIZER,
            "EPOCHS": EPOCHS,
            "TRAIN_SIZE": TRAIN_SIZE,
            "SHUFFLE_BUFFER_SIZE": SHUFFLE_BUFFER_SIZE,
        }, 
        f,
        indent = 4
    )

# save model weights for later usage
model.save_weights(experiment_dir + "/model_weights")

# evaluate model and save metrics:
evaluation = evaluate(model, train_batches, dev_batches)

with open(experiment_dir + "/results.json", "w") as f:
    json.dump(evaluation, f, indent = 4)
