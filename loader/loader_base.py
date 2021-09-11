# -*- coding: utf-8 -*-
"""
Define the loader's parent class
"""
import tensorflow as tf

from util.mode_key import ModeKeys


__all__ = ["DataLoader"]


class DataLoader(object):
    """
    Abstract class for all data loaders

    The sub-classes are expected to implement the abstract default value getters and
    data parsers for each mode
    """
    def __init__(self, source, arch_config, mode=ModeKeys.TRAIN, shuffle=2000, batch_size=128,
                 prefetch=10000, parallel_calls=4, repeat=1):
        self._mode = mode
        self._source = source
        self._shuffle = shuffle
        self._batch_size = batch_size
        self._prefetch = prefetch
        self._parallel_calls = parallel_calls
        self._repeat = repeat
        self._arch_config = arch_config

    def _train_data_parser(self, *fields):
        raise NotImplementedError

    def _train_data_defaults(self):
        raise NotImplementedError

    def _predict_data_parser(self, *fields):
        raise NotImplementedError

    def _predict_data_defaults(self):
        raise NotImplementedError

    def _deploy_data_parser(self, *fields):
        raise NotImplementedError

    def _deploy_data_defaults(self):
        raise NotImplementedError

    def required_fields(self):
        raise NotImplementedError

    def input_fn(self):
        with tf.device("/cpu:0"):
            # Open dataset
            dataset = self._source.open(
                {
                    ModeKeys.TRAIN: self._train_data_defaults,
                    ModeKeys.EVAL: self._train_data_defaults,
                    ModeKeys.PREDICT: self._predict_data_defaults,
                    ModeKeys.DEPLOY: self._deploy_data_defaults
                }[self._mode](),
                field_names=self.required_fields()
            )

            # Parse the fields
            dataset = dataset.map(
                {
                    ModeKeys.TRAIN: self._train_data_parser,
                    ModeKeys.EVAL: self._train_data_parser,
                    ModeKeys.PREDICT: self._predict_data_parser,
                    ModeKeys.DEPLOY: self._deploy_data_parser
                }[self._mode], num_parallel_calls=self._parallel_calls
            )

            # Do shuffle if necessary
            if self._shuffle > 0 and self._mode is ModeKeys.TRAIN:
                dataset = dataset.shuffle(self._shuffle)

            # Do repeat if necessary
            if self._repeat != 1:
                dataset = dataset.repeat(self._repeat)

            # Open the buffer and batch the data items
            dataset = dataset.prefetch(self._prefetch)
            dataset = dataset.batch(self._batch_size)
            return dataset
