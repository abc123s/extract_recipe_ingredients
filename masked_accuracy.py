import tensorflow as tf
from tensorflow import keras

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops

"""
Custom accuracy metric: standard keras model accuracy metric does not 
mask zeros in padded data, leading to an incorrect accuracy.
"""

def sparse_categorical_accuracy_mask_zeros(y_true, y_pred):
    """Calculates how often predictions matches integer labels, with
    zero labels masked.
    You can provide logits of classes as `y_pred`, since argmax of
    logits and probabilities are same.
    Args:
        y_true: Integer ground truth values.
        y_pred: The prediction values.
    Returns:
        Sparse categorical accuracy values.
    """
    y_pred_rank = ops.convert_to_tensor_v2(y_pred).shape.ndims
    y_true_rank = ops.convert_to_tensor_v2(y_true).shape.ndims
    # If the shape of y_true is (num_samples, 1), squeeze to (num_samples,)
    if (y_true_rank is not None) and (y_pred_rank is not None) and (len(
        K.int_shape(y_true)) == len(K.int_shape(y_pred))):
        y_true = array_ops.squeeze(y_true, [-1])
    y_pred = math_ops.argmax(y_pred, axis=-1)

    # If the predicted output and actual output types don't match, force cast them
    # to match.
    if K.dtype(y_pred) != K.dtype(y_true):
        y_pred = math_ops.cast(y_pred, K.dtype(y_true))

    # apply zero mask
    mask = tf.math.not_equal(y_true, 0)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)

    return math_ops.cast(math_ops.equal(y_true_masked, y_pred_masked), K.floatx())

class SparseCategoricalAccuracyMaskZeros(keras.metrics.SparseCategoricalAccuracy):
  """Calculates how often predictions matches integer labels.
  You can provide logits of classes as `y_pred`, since argmax of
  logits and probabilities are same.
  This metric creates two local variables, `total` and `count` that are used to
  compute the frequency with which `y_pred` matches `y_true`. This frequency is
  ultimately returned as `sparse categorical accuracy`: an idempotent operation
  that simply divides `total` by `count`.
  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.
  Usage:
  >>> m = tf.keras.metrics.SparseCategoricalAccuracy()
  >>> _ = m.update_state([[2], [1]], [[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
  >>> m.result().numpy()
  0.5
  >>> m.reset_states()
  >>> _ = m.update_state([[2], [1]], [[0.1, 0.9, 0.8], [0.05, 0.95, 0]],
  ...                    sample_weight=[0.7, 0.3])
  >>> m.result().numpy()
  0.3
  Usage with tf.keras API:
  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile(
      'sgd',
      loss='mse',
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  ```
  """

  def __init__(self, name='sparse_categorical_accuracy_mask_zeros', dtype=None):
    super(keras.metrics.SparseCategoricalAccuracy, self).__init__(
        sparse_categorical_accuracy_mask_zeros, name, dtype=dtype)
