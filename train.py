import tensorflow as tf
from tensorflow import keras

import numpy as np

from preprocess import preprocess
from masked_accuracy import SparseCategoricalAccuracyMaskZeros

train_data, dev_data, _, word_encoder, tag_encoder = preprocess("./data")
BUFFER_SIZE = 1000
train_batches = (train_data
    .take(5000)
    .shuffle(BUFFER_SIZE)
    .padded_batch(32, padded_shapes = ([None], [None]))
)

dev_batches = dev_data.take(1000).padded_batch(32, padded_shapes = ([None], [None]))

model = keras.Sequential([
    keras.layers.Embedding(word_encoder.vocab_size, 64, mask_zero = True),
    keras.layers.LSTM(64, return_sequences=True),
    keras.layers.Dense(tag_encoder.vocab_size),
])

model.summary()

model.compile(
    optimizer="adam",
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics=[SparseCategoricalAccuracyMaskZeros(name = "masked_accuracy"), 'accuracy']
)

history = model.fit(
    train_batches,
    epochs=10,
    validation_data=dev_batches,
    validation_steps=30,
)

loss, masked_accuracy, accuracy = model.evaluate(dev_batches)

print("Loss: ", loss)
print("Masked Accuracy: ", masked_accuracy)
print("Accuracy: ", accuracy)

predict_model = keras.Sequential([
    model,
    keras.layers.Softmax(),
])

tag_correct = 0
tag_total = 0
sentence_correct = 0
sentence_total = 0
for dev_example, dev_label in dev_data.take(1000):
    prediction = keras.backend.flatten(
        keras.backend.argmax(
            model(np.array([dev_example]))
        )
    ).numpy().tolist()
    answer = dev_label.numpy().tolist()

    correct = [pred == label for pred, label in zip(prediction, answer)]
    tag_correct += sum(correct)
    tag_total += len(correct)

    sentence_total += 1
    if len(correct) == sum(correct):
        sentence_correct += 1
    
    if sentence_total % 100 == 0:
        print('')
        print(word_encoder.decode(dev_example))
        print(
            tag_encoder.decode(keras.backend.flatten(
                keras.backend.argmax(
                    model(np.array([dev_example]))
                )
            ))
        )
        print(tag_encoder.decode(dev_label))
        print('summary stats:')
        print(tag_correct)
        print(tag_total)
        print(sentence_correct)
        print(sentence_total)

print('Sentence accuracy')
print(sentence_correct / sentence_total)
print('Tag accuracy')
print(tag_correct / tag_total)

