# -*- coding: utf-8 -*-
import copy

from . import feature_config


class ArchitectureConfig(object):
    """
    Model architecture configuration, including
      * feature_config: Inputs and their encoder configurations
      * model_config: Model structure and related hyper-parameters
    """
    def __init__(self, features, model, raw_data):
        self._feature_conf = features
        self._model_conf = model
        self._raw_data = copy.deepcopy(raw_data)

    @property
    def raw_data(self):
        return self._raw_data

    @property
    def feature_config(self):
        return self._feature_conf

    @property
    def model_config(self):
        return self._model_conf

    @staticmethod
    def parse(json_object, model_conf_parser):
        raw_data = copy.deepcopy(json_object)
        return ArchitectureConfig(
            features=feature_config.InputFeatureConfig.parse(json_object["features"]),
            model=model_conf_parser(json_object.get("model", {})),
            raw_data=raw_data
        )
