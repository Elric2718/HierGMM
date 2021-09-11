# -*- coding: utf-8 -*-
"""
Complex tensorflow operations for re-use
"""
import six
import tensorflow as tf


try:
    _SparseTensor = tf.sparse.SparseTensor
    _sparse_to_dense = tf.sparse.to_dense
except:
    _SparseTensor = tf.SparseTensor
    _sparse_to_dense = tf.sparse_tensor_to_dense


def sparse_tensor(indices, values, shape, to_dense=False):
    """
    Build sparse tensor
    :param indices: A list contains one or several 1D index tensors.
    :param values: A 1D tensor containing Element values
    :param to_dense: Indicating whether converting to dense shape
    :return:
    """
    def _ensure_index_type(data_index):
        if data_index.dtype is tf.int64 or "int64" in str(data_index.dtype):
            return data_index
        else:
            return tf.cast(data_index, tf.int64)

    indices = [_ensure_index_type(index) for index in indices]
    tensor = _SparseTensor(
        indices=tf.stack(indices, axis=-1),
        values=values,
        dense_shape=shape
    )

    if to_dense:
        return _sparse_to_dense(tensor, validate_indices=False)
    else:
        return tensor


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

    Raises:
    ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
          "For the tensor `%s` in scope `%s`, the actual rank "
          "`%d` (shape = %s) is not equal to the expected rank `%s`" %
          (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

    Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def get_mask(inputs, empty_value=0):
    return tf.cast(tf.not_equal(inputs, empty_value), tf.float32)


def get_length(inputs, masks=None, empty_value=0):
    """
    :param inputs: N-D value matrix. If mask is provided, it is ignored
    :param masks: Binary matrix has the same shape as inputs
    :param empty_value:
    :return:
    """
    if masks is None:
        masks = get_mask(inputs, empty_value)

    return tf.reduce_sum(masks, axis=-1)
