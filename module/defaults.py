# -*- coding: utf-8 -*-
"""
Defines some default settings for model modules
"""
import tensorflow as tf


def weight_initializer():
    return tf.contrib.layers.xavier_initializer()


def bias_initializer():
    return tf.zeros_initializer()


def variable_scope(name):
    return tf.variable_scope(name, reuse=tf.AUTO_REUSE)
