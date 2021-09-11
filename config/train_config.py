# -*- coding: utf-8 -*-
"""
Runtime configurations
"""
from collections import namedtuple
import logging

import tensorflow as tf


class LearningRate(object):
    def __init__(self, learning_rate):
        self._lr_segments = str(learning_rate).split(",")

    def _parse_learning_rate(self):
        pass

    def learning_rate(self, global_step):
        if len(self._lr_segments) == 1:
            # Constant learning rate
            logging.info("Constant learning rate {} is used".format(self._lr_segments[0]))
            return float(self._lr_segments[0])
        elif len(self._lr_segments) == 3:
            init_lr, decay_rate, decay_steps = self._lr_segments
            logging.info(
                "Toggling exponential decay: init_lr={},decay_rate={},decay_steps={}".format(
                    init_lr, decay_rate, decay_steps
                )
            )
            return tf.train.exponential_decay(
                float(init_lr),
                decay_steps=int(decay_steps),
                decay_rate=float(decay_rate),
                staircase=True,
                global_step=global_step,
            )

TrainConfig = namedtuple("TrainConfig", (
    "learning_rate",
    "batch_size",
    "max_gradient_norm",
    "log_every",
    "init_checkpoint"
))
