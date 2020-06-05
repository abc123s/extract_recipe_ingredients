import json

import tensorflow as tf

from preprocess import preprocess
from build_model import build_model
from masked_accuracy import SparseCategoricalAccuracyMaskZeros
from evaluate import evaluate

experiment_dir = "experiments/20200602_2058_df1046b"

# load experiment params
with open(experiment_dir + "/params.json", "r") as f:
    params = json.load(f)

# grab train and dev sets
all_train_data, dev_data, _, word_encoder, tag_encoder = preprocess("./data")

train_data = all_train_data.take(params["TRAIN_SIZE"])

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

train_batches = train_data.padded_batch(128, padded_shapes = ([None], [None]))
dev_batches = dev_data.padded_batch(128, padded_shapes = ([None], [None]))

# reevaluate the model and save metrics
evaluation = evaluate(model, train_batches, dev_batches)

with open(experiment_dir + "/results.json", "w") as f:
    json.dump(evaluation, f, indent = 4)
