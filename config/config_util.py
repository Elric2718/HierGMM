# -*- coding: utf-8 -*-
"""
Some commonly used module config objects
"""
import copy
import pprint


class ModuleConfig(object):
    """
    Configs for a specific neural network module
    """
    def __init__(self, name, type, params):
        """
        :param name: Module name, for debugging only
        :param type: Module type, usually as key for a factory
        :param params: Module params
        """
        self._name = name
        self._type = type
        # If params is None, we assign a empty dictionary
        self._params = copy.deepcopy(params) or dict()
        # self._projection = ProjectionConfig.parse(self._params.get("projection"))

    @property
    def type(self):
        return self._type

    def get(self, param_key, default=None):
        return self._params.get(param_key, default=default)

    @property
    def params(self):
        return self._params

    @property
    def projection(self):
        return

    def __getitem__(self, item):
        return self._params[item]

    def __str__(self):
        return "{}[{}]{}".format(
            self._name,
            self._type,
            pprint.pformat(self._params)
        )

    @staticmethod
    def parse(name, json_object):
        """
        Parse from json config object
        :param name: the module name
        :param json_object: config object parsed from a
        :return:
        """
        type = json_object.pop("type")
        return ModuleConfig(
            name=name,
            type=type,
            params=json_object
        )


# class ProjectionType:
#     FULLY_CONNECTED = "fc"
#     RESIDUAL = "residual"


# class ProjectionConfig(object):
#     """
#     Projection module is used for making the model deeper or dimension conversion
#     """
#     def __init__(self, layers, projection_type=ProjectionType.FULLY_CONNECTED):
#         self._layers = list(layers)
#         self._projection_type = projection_type
#
#     def __iter__(self):
#         return iter(self._layers)
#
#     def __getitem__(self, item):
#         return self._layers[item]
#
#     def __len__(self):
#         return len(self._layers)
#
#     @property
#     def valid(self):
#         return len(self._layers) > 0
#
#     @property
#     def layers(self):
#         return self._layers
#
#     @property
#     def projection_type(self):
#         return self._projection_type
#
#     @staticmethod
#     def parse(json_object):
#         # json_object might be more complicated in future
#         assert isinstance(json_object, string_types) or json_object is None
#         json_object = json_object or ""
#         return ProjectionConfig(
#             [int(n) for n in json_object.split(",") if len(n.strip()) > 0]
#         )
#
#     def __str__(self):
#         return "projection[{}]={}".format(self._projection_type, self._layers)
