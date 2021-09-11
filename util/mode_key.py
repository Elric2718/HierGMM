# -*- coding: utf-8 -*-
"""
Runtime mode keys
"""
import tensorflow as tf


class ModeKeys:
    """
    Extended mode keys
    """
    TRAIN = tf.estimator.ModeKeys.TRAIN

    PREDICT = tf.estimator.ModeKeys.PREDICT

    EVAL = tf.estimator.ModeKeys.EVAL

    # This is an extended mode to indicate the online prediction mode
    DEPLOY = "deploy"
