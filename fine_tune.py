'''
Takes a model trained on the NYT dataset, and fine-tune it using
only manually tagged data, which has a lower error rate
'''

import os
from datetime import datetime
import subprocess
import json

import tensorflow as tf

from preprocess_manual import preprocess, load_examples as load_manual_examples
from build_model import build_model
from masked_accuracy import SparseCategoricalAccuracyMaskZeros
from evaluate import evaluate

from preprocess_original import crfFile2Examples
from preprocess_simple import load_examples as load_simple_examples


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

# previous experiment to fine tune
original_experiment_dir = "experiments/20200607_0646_53eb83b"

# load experiment params
with open(original_experiment_dir + "/params.json", "r") as f:
    old_params = json.load(f)

# params that do not affect the model architecture, and thus
# we can change when fine-tuning
new_params = {
    "EPOCHS": 20,
    "REGULARIZER": None,
    "REGULARIZATION_FACTOR": 0,
    "DROPOUT_RATE": 0,
    "RECURRENT_DROPOUT_RATE": 0,
    "OPTIMIZER": "adam"
}

# make final list of params
params = {
    **old_params,
    **new_params,
}

# grab list of examples used to generate original vocab list
# / word encoder (needed for preprocessing manually tagged data)
original_examples = example_loader[params.get("PREPROCESSOR", "original")]()

# grab train and dev sets to do fine-tuning on
train_data, dev_data, _, word_encoder, tag_encoder = preprocess(
    "./data", original_examples)

train_batches = train_data.padded_batch(128, padded_shapes=([None], [None]))
dev_batches = dev_data.padded_batch(128, padded_shapes=([None], [None]))

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

# load final weights from original experiment into model:
model.load_weights(original_experiment_dir + "/model_weights")

# make new experiment directory and save experiment params down
date_string = datetime.now().strftime("%Y%m%d_%H%M")
commit_string = subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
experiment_dir = "experiments/" + date_string + "_" + commit_string
os.mkdir(experiment_dir)

# add tensorboard logs
epoch_tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=experiment_dir + "/epoch_logs", histogram_freq=1)

# fine-tune model:
history = model.fit(train_batches,
                    epochs=new_params["EPOCHS"],
                    validation_data=dev_batches,
                    validation_steps=40,
                    callbacks=[epoch_tensorboard_callback])

# save params down
with open(experiment_dir + "/params.json", "w") as f:
    json.dump({
        **params,
        "ORIGINAL_EXPERIMENT_DIR": original_experiment_dir,
    },
              f,
              indent=4)

# save model weights for later usage
model.save_weights(experiment_dir + "/model_weights")

# evaluate model and save metrics:
evaluation = evaluate(model, train_batches, dev_batches,
                      params["PREPROCESSOR"] == 'combined')

with open(experiment_dir + "/results.json", "w") as f:
    json.dump(evaluation, f, indent=4)
