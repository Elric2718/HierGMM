# -*- coding: utf-8 -*-
"""
Projector modules
"""
import tensorflow as tf

from . import defaults
from . import normalization


def _ln(inputs, epsilon=1e-8, scope="ln"):
    """Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def _residual_block(inputs, num_units, scope):
    '''position-wise feed forward net

    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Inner layer
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

        # Outer layer
        outputs = tf.layers.dense(outputs, num_units[1])

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = _ln(outputs)

    return outputs


def multi_layer_perceptron(inputs, layers, activation_fn, scope_name, layer_norm=False, mask=None, residual=False, *args, **kwargs):
    values = inputs
    with defaults.variable_scope(scope_name):
        for i, n_units in enumerate(layers, 1):
            with defaults.variable_scope("mlp_layer_%d" % i):
                if residual:
                    values = _residual_block(inputs, num_units=[3*n_units, n_units], scope="residual")
                else:
                    values = tf.layers.dense(
                        inputs=values,
                        units=n_units,
                        activation=activation_fn,
                        kernel_initializer=defaults.weight_initializer()
                    )

                    if layer_norm:
                        values = normalization.layer_norm(values, scope_name="projection_norm")

                if mask is not None:
                    values = tf.multiply(tf.expand_dims(mask, axis=-1), values)
                    print("SeqMask:", mask)

    return values
