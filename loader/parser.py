# -*- coding: utf-8 -*-
"""
Data field parser functions, used to facilitate data field content parsing
"""
import tensorflow as tf


def split_text(text, delimiter=" ", shape=None, limit=None, ):
    """
    Split the text by specified delimiter
    :param text: Input 1D text Tensor
    :param delimiter: Value delimiter
    :param shape: If set, converted values would be reshaped
    :param limit: If set, slicing would be conducted to restrict the maximal limit
    :return: An 1D Tensor containing converted strings
    """
    values = tf.string_split([text], delimiter).values

    # Do truncation
    if limit:
        values = values[:limit]

    if shape:
        values = tf.reshape(values, shape)

    return values


def split_and_convert(text, out_type, delimiter=" ", shape=None, limit=None, pad_to=None, padding_value=0, op_name=None):
    """
    Split the text by specified delimiter and then convert values into the given numerical type
    :param text: Input 1D text Tensor
    :param out_type: Target data types like tf.int32, tf.float32, etc
    :param delimiter: Value delimiter
    :param shape: If set, converted values would be reshaped
    :param limit: If set, slicing would be conducted to restrict the maximal limit
    :param pad_to: Int value. If set, returned tensor would be padded to the shape specified by this value
    :param padding_value: A number. If set, returned tensor would be padded with this value
    :return: An 1D Tensor containing converted values
    """
    values = tf.string_to_number(
        split_text(text, delimiter, None, limit),
        out_type=out_type,
        name="{}_s2i".format(op_name)
    )

    if pad_to is not None:
        values = tf.pad(
            values, [[0, pad_to - tf.shape(values)[-1]]], "CONSTANT", constant_values=padding_value,
            name="{}_padding".format(op_name)
        )
        values = tf.reshape(values, [pad_to])

    if shape:
        values = tf.reshape(values, shape)

    return values


def decode_base64(data, out_type, shape=None, limit=None, pad_to=None, padding_value=0):
    """
    Parse base64-encoded string as values
    :param data: 1D string Tensor containing base64-encoded data
    :param out_type: Target data types like tf.int32, tf.float32, etc
    :param shape: If set, converted values would be reshaped
    :param limit: If set, slicing would be conducted to restrict the maximal limit
        :param pad_to: Int value. If set, returned tensor would be padded to the shape specified by this value
    :param padding_value: A number. If set, returned tensor would be padded with this value
    :return: A 1D Tensor containing converted values
    """
    values = tf.decode_raw(tf.decode_base64(data), out_type=out_type)
    # Do truncation
    if limit:
        values = values[:limit]

    if shape and not pad_to:
        values = tf.reshape(values, shape)

    if pad_to is not None:
        values = tf.pad(values, [[0, pad_to - tf.shape(values)[-1]]], "CONSTANT", constant_values=padding_value)
        values = tf.reshape(values, [pad_to])

    if shape is not None:
        values = tf.reshape(values, shape)

    return values
