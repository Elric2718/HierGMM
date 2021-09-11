import abc
import six
import tensorflow as tf


@six.add_metaclass(abc.ABCMeta)
class AttentionLayer(object):

    @abc.abstractmethod
    def score_fn(self, keys, query, num_units):
        """Computes the attention score"""
        raise NotImplementedError

    def _build(self, query, keys, values, values_length, num_units):
        """Computes attention scores and outputs.
        Args:
          query: The query used to calculate attention scores.
            In abstractive_summay this is typically the current state of the decoder.
            A tensor of shape `[B, ...]`
          keys: The keys used to calculate attention scores. In abstractive_summay, these
            are typically the outputs of the encoder and equivalent to `values`.
            A tensor of shape `[B, T, ...]` where each element in the `T`
            dimension corresponds to the key for that value.
          values: The elements to compute attention over. In abstractive_summay, this is
            typically the sequence of encoder outputs.
            A tensor of shape `[B, T, input_dim]`.
          values_length: An int32 tensor of shape `[B]` defining the sequence
            length of the attention values.

        Returns:
          A tuple `(scores, context)`.
          `scores` is vector of length `T` where each element is the
          normalized "score" of the corresponding `inputs` element.
          `context` is the final attention layer output corresponding to
          the weighted inputs.
          A tensor fo shape `[B, input_dim]`.
        """
        values_depth = values.get_shape().as_list()[-1]

        # Fully connected layers to transform both keys and query
        # into a tensor with `num_units` units
        att_keys = tf.layers.dense(inputs=keys,
                                   units=num_units,
                                   activation=None,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())
        att_query = tf.layers.dense(inputs=query,
                                    units=num_units,
                                    activation=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())

        scores = self.score_fn(att_keys, att_query, num_units)

        # Replace all scores for padded inputs with tf.float32.min
        num_scores = tf.shape(scores)[1]
        scores_mask = tf.sequence_mask(lengths=values_length,
                                       maxlen=num_scores,
                                       dtype=tf.float32)
        scores = scores * scores_mask + ((1.0 - scores_mask) * -64.0)
        scores = scores -  tf.tile(tf.reduce_max(scores, axis=1, keep_dims=True), [1, num_scores])

        # Normalize the scores
        scores_normalized = tf.nn.softmax(scores, name="scores_normalized")

        # Calculate the weighted average of the attention inputs
        # according to the scores
        context = tf.expand_dims(scores_normalized, 2) * values
        context_ = tf.reduce_sum(input_tensor=context, axis=1, name="context")
        context_.set_shape([None, values_depth])

        return context_


class AttentionLayerDot(AttentionLayer):
    """
    An attention layer that calculates attention scores using a dot product.
    returns:
        shape is [B, T]
    """
    def score_fn(self, keys, query, num_units):
        return tf.reduce_sum(keys * tf.expand_dims(query, 1), [2])


class AttentionLayerBahdanau(AttentionLayer):
    """
    An attention layer that calculates attention scores using a parameterized multiplication.
    returns:
        shape is [B, T]
    """
    def score_fn(self, keys, query, num_units):
        v_att = tf.get_variable(
            "v_att", shape=[num_units], dtype=tf.float32)
        return tf.reduce_sum(v_att * tf.tanh(keys + tf.expand_dims(query, 1)), [2])

class MultiHeadAttentionLayer(AttentionLayer):
    """
    An attention layer that calculates attention scores using a parameterized multiplication.
    returns:
        shape is [B, T]
    """
    '''attention in transformer (multiplicative attention)'''
    def score_fn(self, keys, query, num_units):
        return tf.reduce_sum(keys * query, [2]) / tf.sqrt(tf.cast(num_units, tf.float32))

    def _build(self, query, keys, values, values_length, num_units, num_heads=8, scope='multihead_attention', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            Q = tf.layers.dense(tf.expand_dims(query, 1), num_units * num_heads, use_bias=True, activation=None) # N x 1 x C   (C = num_units * num_heads)
            K = tf.layers.dense(keys, num_units * num_heads, use_bias=True, activation=None) # N x M x C
            V = tf.layers.dense(values, num_units * num_heads, use_bias=True, activation=None) # N x M x C

            # split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # hN x 1 x C/h
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # hN x M x C/h
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # hN x M x C/h

            # compute the logit score
            scores = self.score_fn(K_, Q_, num_units)     # hN x M

            num_scores = tf.shape(scores)[1]
            scores_mask = tf.sequence_mask(lengths=values_length,
                                           maxlen=num_scores,
                                           dtype=tf.float32)  # N x M
            scores_mask = tf.tile(scores_mask, [num_heads, 1]) # hN x M
            scores = scores * scores_mask + ((1.0 - scores_mask) * -(2**32.0))
            scores = scores - tf.tile(tf.reduce_max(scores, axis=1, keep_dims=True), [1, num_scores])

            # Normalize the scores
            scores_normalized = tf.nn.softmax(scores, name="scores_normalized")  # hN x M
            outputs = tf.matmul(tf.expand_dims(scores_normalized, 1), V_) # hN x 1 x C/h
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2) # N x 1 x C

            outputs = tf.squeeze(outputs, axis=1)   # N x C

            return outputs

class MultiHeadAttentionLayerWithMute(AttentionLayer):
    """
    An attention layer that calculates attention scores using a parameterized multiplication.
    returns:
        shape is [B, T]
    if value_length = 0 for one given key, mute the output, i.e., the corresponding output is zero
    """
    '''attention in transformer (multiplicative attention)'''
    def score_fn(self, keys, query, num_units):
        return tf.reduce_sum(keys * query, [2]) / tf.sqrt(tf.cast(num_units, tf.float32))

    def _build(self, query, keys, values, values_length, num_units, num_heads=8, scope='multihead_attention', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            Q = tf.layers.dense(tf.expand_dims(query, 1), num_units * num_heads, use_bias=True, activation=None) # N x 1 x C   (C = num_units * num_heads)
            K = tf.layers.dense(keys, num_units * num_heads, use_bias=True, activation=None) # N x M x C
            V = tf.layers.dense(values, num_units * num_heads, use_bias=True, activation=None) # N x M x C

            # split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # hN x 1 x C/h
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # hN x M x C/h
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # hN x M x C/h

            # compute the logit score
            scores = self.score_fn(K_, Q_, num_units)     # hN x M

            num_scores = tf.shape(scores)[1]
            scores_mask = tf.sequence_mask(lengths=values_length,
                                           maxlen=num_scores,
                                           dtype=tf.float32)  # N x M
            scores_mask = tf.tile(scores_mask, [num_heads, 1]) # hN x M
            scores = scores * scores_mask + ((1.0 - scores_mask) * -(2**32.0))
            scores = scores - tf.tile(tf.reduce_max(scores, axis=1, keep_dims=True), [1, num_scores])

            # Normalize the scores
            scores_normalized = tf.nn.softmax(scores, name="scores_normalized")  # hN x M
            outputs = tf.matmul(tf.expand_dims(scores_normalized, 1), V_) # hN x 1 x C/h
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2) # N x 1 x C

            outputs = tf.squeeze(outputs, axis=1)   # N x C

            # mute the output if value_length = 0
            mute_mask = tf.where(tf.equal(values_length, 0), tf.zeros_like(values_length, dtype=tf.float32), tf.ones_like(values_length, dtype=tf.float32)) # N
            mute_mask = tf.expand_dims(mute_mask, 1) # N x 1
            outputs = outputs * mute_mask

            return outputs
