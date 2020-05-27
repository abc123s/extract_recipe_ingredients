from tensorflow import keras
import numpy as np

def sentence_level_accuracy(model, data):
    sentence_correct = 0
    sentence_total = 0
    for example, label in data:
        prediction = keras.backend.flatten(
            keras.backend.argmax(
                model(np.array([example]))
            )
        ).numpy().tolist()
        answer = label.numpy().tolist()

        correct = [pred == label for pred, label in zip(prediction, answer)]

        sentence_total += 1
        if len(correct) == sum(correct):
            sentence_correct += 1

    sentence_accuracy = sentence_correct / sentence_total

    return sentence_correct, sentence_total, sentence_accuracy

def tag_level_accuracy(model, data_batches):
    loss, tag_accuracy = model.evaluate(data_batches)

    return tag_accuracy


def accuracy(model, data):
    # batch data for tag level accuracy conputation
    batches = data.padded_batch(128, padded_shapes = ([None], [None]))

    # compute tag accuracy
    tag_accuracy = tag_level_accuracy(model, batches)

    # compute sentence accuracy
    sentence_correct, sentence_total, sentence_accuracy = sentence_level_accuracy(model, data)

    return {
        "Tag-Level Stats": {
            "Accuracy": tag_accuracy,
        },
        "Sentence-Level Stats": {
            "Correct": sentence_correct,
            "Total": sentence_total,
            "Accuracy": sentence_accuracy,
        },
    }

def evaluate(model, train_data, dev_data):
    return {
        "Train": accuracy(model, train_data),
        "Dev": accuracy(model, dev_data)
    }