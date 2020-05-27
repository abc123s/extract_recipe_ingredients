from tensorflow import keras
import numpy as np

def sentence_level_accuracy(model, data_batches):
    sentence_correct = 0
    sentence_total = 0

    for examples, labels in data_batches:
        model_outputs = model.predict(examples)
        for model_output, label in zip(model_outputs, labels):
            prediction = keras.backend.flatten(
                keras.backend.argmax(
                    model_output
                )
            ).numpy().tolist()

            answer = label.numpy().tolist()

            correct = [pred == label or label == 0 for pred, label in zip(prediction, answer)]

            sentence_total += 1
            if len(correct) == sum(correct):
                sentence_correct += 1
    
    sentence_accuracy = sentence_correct / sentence_total

    return sentence_correct, sentence_total, sentence_accuracy

def tag_level_accuracy(model, data_batches):
    loss, tag_accuracy = model.evaluate(data_batches)

    return tag_accuracy


def accuracy(model, batches):
    # compute tag accuracy
    tag_accuracy = tag_level_accuracy(model, batches)

    # compute sentence accuracy
    sentence_correct, sentence_total, sentence_accuracy = sentence_level_accuracy(model, batches)

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

def evaluate(model, train_batches, dev_batches):
    return {
        "Train": accuracy(model, train_batches),
        "Dev": accuracy(model, dev_batches)
    }