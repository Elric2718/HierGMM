# -*- coding: utf-8 -*-
"""
This script defines modules for multi embedding aggregation
"""
import tensorflow as tf

from util import tensor_op
from util import control_flow
from . import defaults


_aggregator = control_flow.Register()
aggregate = _aggregator.build
get_aggregator = _aggregator.get


def _stack_tensors(inputs):
    if inputs is None or type(inputs) not in [tuple, list]:
        return inputs
    else:
        return tf.stack(inputs, axis=-2)


@_aggregator("mean")
def mean_pooling(inputs, mask=None, length=None, *args, **kwargs):
    """
    Perform average pooling on inputs
    :param inputs: (N)-D tensor. The second to last dimension would be pooled
    :param mask: (N-1)-D binary int tensor. If set, the mask would be multiplied to inputs
    :param length: (N-2)-D int tensor. If not set, the length would be inferred from mask
    :return:
    """
    inputs, mask = _stack_tensors(inputs), _stack_tensors(mask)
    if type(inputs) is list:
        inputs = tf.concat(inputs, axis=-2)

    if length is None and mask is not None:
        length = tensor_op.get_length(None, mask)

    if mask is not None:
        inputs = tf.multiply(inputs, tf.expand_dims(mask, axis=-1))

    if length is None:
        return tf.reduce_mean(inputs, axis=-2)
    else:
        inputs = tf.reduce_sum(inputs, axis=-2)
        return tf.div(inputs, tf.maximum(1.0, tf.expand_dims(length, axis=-1)))


@_aggregator("sum")
def sum_pooling(inputs, mask=None, *args, **kwargs):
    """
    Perform summation pooling on inputs
    :param inputs: (N)-D tensor. The second to last dimension would be pooled
    :param mask: (N-1)-D binary int tensor. If set, the mask would be multiplied to inputs
    :return:
    """
    inputs, mask = _stack_tensors(inputs), _stack_tensors(mask)
    if mask is not None:
        inputs = tf.multiply(inputs, tf.expand_dims(mask, axis=-1))
    return tf.reduce_sum(inputs, axis=-2)


@_aggregator("max")
def max_pooling(inputs, mask=None, scope_name=None, trainable_empty_tensor=True, *args, **kwargs):
    """
    Perform max pooling on inputs
    :param inputs: (N)-D tensor. The second to last dimension would be pooled
    :param mask: (N-1)-D binary int tensor. If set, the mask would be multiplied to inputs
    :param scope_name: Variable scope name. If trainable_empty_tensor is set, it can be None
    :param trainable_empty_tensor: If set, a free tensor is allocated to represent an all-masked empty embedding
    :return:
    """
    inputs, mask = _stack_tensors(inputs), _stack_tensors(mask)

    if mask is not None:
        inputs += tf.expand_dims(-1e30 * (1 - mask), axis=-1)

    result = tf.reduce_max(inputs, axis=-2)
    if trainable_empty_tensor:
        with defaults.variable_scope(scope_name):
            shape = [1] * (len(result.shape) - 1) + [int(result.shape[-1])]
            empty_embedding = tf.get_variable(
                name="empty_embedding", dtype=tf.float32, shape=shape,
                initializer=defaults.weight_initializer()
            )
            return tf.maximum(result, empty_embedding)
    else:
        return result


@_aggregator("concat")
def concatenate(inputs, *args, **kwargs):
    """
    Concat multiple embeddings.
    :param inputs: Single (N)-D tensor or a list of (N)-D tensors.
           For the former case, the second-to-last dimension is unstacked and then concatenated
           For the latter case, tf.concat is directly applied to concat the dimensions
    :return:
    """
    if type(inputs) is list:
        return tf.concat(inputs, axis=-1)
    else:
        return tf.concat(tf.unstack(inputs, axis=-2), axis=-1)


@_aggregator("cnn")
def convolution(inputs, kernel, layers=1, scope_name=None, pooling=True):
    """
    Apply convolution along the second-to-last dimension with the kernel width equals to
    to the last dimension.
    :param inputs: N-D tensor, N >= 2
    :param kernel: A string representing kernel configuration. For example 2,128,3,128
    :param layers: Int value indicating number of repetitions of convolution
    :param scope_name: Variable scope name
    :param pooling: Boolean value. If set true, max pooling would be performed
    :return:
    """
    raise NotImplementedError
