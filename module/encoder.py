# -*- coding: utf-8 -*-
"""
Utility encoder modules
"""
from collections import defaultdict

import numpy as np
import tensorflow as tf

from config import FeatureField
from module import defaults
from util import control_flow
from module import projector
from module import aggregator


_encoder = control_flow.Register()


__all__ = ["encode_input", "EncoderContext"]


def encode_input(input_tensor, encoding_conf, scope_name, context=None):
    """
    Encode input as embedding
    :param input_tensor: Tensor with shape B*L*D
    :param encoding_conf: Encoding configuration
    :type encoding_conf: FeatureField
    :param scope_name: str, usually set to domain name
    :return:
    """
    with defaults.variable_scope(scope_name):
        return _encoder.get(encoding_conf.transform.type)(
            inputs=input_tensor,
            scope_name=encoding_conf.name,
            context=context,
            **encoding_conf.transform.params
        )


@_encoder("lookup")
def embedding_lookup(inputs, vocab_size=0, dimension=0, padding=0., scope_name=None, context=None, shared=None,
                     squeeze_input=True, *args, **kwargs):
    """
    Retrieve a free embedding for the last dimension
    :param inputs: B*L*1 int matrix. Each element corresponds to an id
    :param vocab_size: Vocabulary size
    :param dimension: Embedding dimension
    :param padding: If not None, a constant embedding would be padded to the embedding matrix
    :param scope_name:
    :return:
    """
    if shared and not context:
        raise ValueError("Context is set to None as shared is toggled")

    shared_matrices = context["encoder.embedding_lookup"] if context is not None else dict()

    with defaults.variable_scope(scope_name):
        if shared and shared in shared_matrices:
            embedding_matrix = shared_matrices[shared]
        else:
            embedding_matrix = tf.get_variable(
                "embedding_matrix",
                [vocab_size + int(padding is None), dimension],
                dtype=tf.float32,
                initializer=defaults.weight_initializer()
            )

            if padding is not None:
                constant_embedding = tf.constant(
                    np.array([padding] * dimension, dtype=np.float32).reshape([1, -1]),
                    dtype=tf.float32
                )
                embedding_matrix = tf.concat([constant_embedding, embedding_matrix], axis=0)

            if shared:
                shared_matrices[shared] = embedding_matrix

        if squeeze_input:
            inputs = tf.squeeze(inputs, axis=2)

        return tf.gather(embedding_matrix, inputs)


@_encoder("mean_pooling")
def mean_pooling_encoder(inputs, vocab_size=0, dimension=0, scope_name=None, *args, **kwargs):
    """
    Retrieve a free embedding for the last dimension and then perform average pooling
    :param inputs:
    :param mask:
    :param vocab_size:
    :param dimension:
    :param scope_name:
    :param args:
    :param kwargs:
    :return:
    """
    token_embeddings = embedding_lookup(inputs, vocab_size, dimension, 0., scope_name, squeeze_input=False,
                                        *args, **kwargs)
    return aggregator.mean_pooling(token_embeddings)


@_encoder("identity")
def identity(inputs, *args, **kwargs):
    """
    Perform an identity transformation
    :return:
    """
    return tf.cast(inputs, tf.float32)


@_encoder("projection")
def projection(inputs, layers, scope_name, **kwargs):
    """
    Perform one or several dense projection operation
    :return:
    """
    return projector.multi_layer_perceptron(
        inputs=inputs,
        layers=layers,
        activation_fn=tf.nn.tanh,
        scope_name=scope_name
    )

@_encoder("cnn")
def convolution_encoder():
    raise NotImplementedError


class EncoderContext(object):
    def __init__(self):
        self._data = defaultdict(dict)

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, key, value):
        self._data[key] = value

    def get(self, key, default):
        return self._data.get(key, default)

    def set(self, key, value):
        self._data[key] = value
