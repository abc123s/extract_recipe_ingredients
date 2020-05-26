import os
from datetime import datetime
import subprocess

import numpy as np

import tensorflow as tf
from tensorflow import keras

from preprocess import preprocess
from build_model import build_model
from masked_accuracy import SparseCategoricalAccuracyMaskZeros

train_data, dev_data, _, word_encoder, tag_encoder = preprocess("./data")

TRAIN_SIZE = 169207
BUFFER_SIZE = 1000
train_batches = (train_data
    .take(TRAIN_SIZE)
    .shuffle(BUFFER_SIZE)
    .padded_batch(128, padded_shapes = ([None], [None]))
)

dev_batches = dev_data.padded_batch(128, padded_shapes = ([None], [None]))

# build model:
ARCHITECTURE = "rnn"
model = build_model(ARCHITECTURE, word_encoder.vocab_size, tag_encoder.vocab_size)

# compile model:
OPTIMIZER = "adam"
model.compile(
    optimizer=OPTIMIZER,
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics=[SparseCategoricalAccuracyMaskZeros(name = "accuracy")]
)

# fit model:
EPOCHS = 10
history = model.fit(
    train_batches,
    epochs = EPOCHS,
    validation_data=dev_batches,
    validation_steps=8,
)

# evaluate model:

# Tag-level accuracy and loss
loss, tag_accuracy = model.evaluate(dev_batches)


# Sentence-level accuracy
sentence_correct = 0
sentence_total = 0
for dev_example, dev_label in dev_data:
    prediction = keras.backend.flatten(
        keras.backend.argmax(
            model(np.array([dev_example]))
        )
    ).numpy().tolist()
    answer = dev_label.numpy().tolist()

    correct = [pred == label for pred, label in zip(prediction, answer)]

    sentence_total += 1
    if len(correct) == sum(correct):
        sentence_correct += 1

sentence_accuracy = sentence_correct / sentence_total

# save experiment results down
date_string = datetime.now().strftime("%Y%m%d_%H%M")
commit_string = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
experiment_dir = "experiments/" + date_string + "_" + commit_string
os.mkdir(experiment_dir)

# save model weights for later usage
model.save_weights(experiment_dir + "/model_weights")

# save params and results
f = open(experiment_dir + "/results.txt", "w")

f.write("Params\n")
f.write("ARCHITECTURE: " + ARCHITECTURE + "\n")
f.write("OPTIMIZER: " + OPTIMIZER + "\n")
f.write("EPOCHS: " + str(EPOCHS) + "\n")
f.write("TRAIN_SIZE: " + str(TRAIN_SIZE) + "\n\n")

f.write("Sentence-Level Stats:" + "\n")
f.write("\tcorrect: " + str(sentence_correct) + "\n")
f.write("\ttotal: " + str(sentence_total) + "\n")
f.write("\t% correct: " + str(100 * sentence_accuracy) + "\n\n")

f.write("Word-Level Stats:\n")
f.write("\t% correct:" + str(100 * tag_accuracy) + "\n")

f.close()

