# -*- coding: utf-8 -*-
"""
Normalization modules
"""
import tensorflow as tf

from . import defaults


def layer_norm(inputs, scope_name, units=None, activation_fn=None, scale_and_center=True, *args, **kwargs):
    """
    Perform layer normalization to the last dimension of inputs
    :param inputs: Input tensor
    :param scope_name: Variable scope name
    :param units: If set, a fully connected layer would be conducted before layer norm
    :param activation_fn: The activation_fn used in the fully connected layer mentioned above
    :return:
    """
    with defaults.variable_scope(scope_name):
        if units is not None:
            inputs = tf.layers.dense(
                inputs=inputs,
                units=units,
                activation=activation_fn,
                kernel_initializer=defaults.weight_initializer()
            )
        return tf.contrib.layers.layer_norm(
            inputs,
            begin_norm_axis=-1,
            scale=scale_and_center,
            center=scale_and_center
        )
