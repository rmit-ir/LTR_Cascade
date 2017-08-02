"""
Core helper functions
"""
import numpy as np


def polarize(v):
    return (v > 0).astype(int) * 2 - 1


def get_class_weights(y):
    return np.transpose(np.unique(y))


def get_score(predictions, class_weights):
    """Convert predictions to scores."""
    return predictions


def get_score_multiclass(predictions, class_weights):
    """Convert predictions (for multiclass classifier) to scores.

    Score for multi-class is sum class_i * p(class_i)
    :param predictions:
    :param class_weights:
    :return:
    """
    new_predictions = np.zeros(shape=(predictions.shape[0]))
    for i in range(0, predictions.shape[0]):
        j = np.dot(predictions[i], class_weights)
        new_predictions[i] = j
    return new_predictions
