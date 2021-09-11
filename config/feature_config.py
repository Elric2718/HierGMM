# -*- coding: utf-8 -*-del
import enum
import pprint

import tensorflow as tf

from . import config_util as cu


_PROPERTY_NAME_SEPARATOR = "."


class FeatureFormat(enum.Enum):
    TEXT = "text"
    BASE64 = "base64"
    COMPACT = "compact"


class FeatureField(object):
    def __init__(self, domain_name, name, dtype, dimension, format, transform, params):
        self._name = name
        self._dtype = dtype
        self._format = format
        self._transform = transform
        self._domain_name = domain_name
        self._dimension = dimension
        self._others = dict()
        self._params = params

    @property
    def name(self):
        return self._name

    @property
    def full_name(self):
        return self._domain_name + _PROPERTY_NAME_SEPARATOR + self._name

    @property
    def dtype(self):
        return self._dtype

    @property
    def dimension(self):
        return self._dimension

    @property
    def format(self):
        return self._format

    @property
    def transform(self):
        return self._transform

    def set(self, name, value):
        self._others[name] = value

    def get(self, name, default_value=None):
        return self._others.get(name, default_value)

    def param(self, param_name, default_value=None):
        return self._params.get(param_name, default_value)

    @staticmethod
    def parse(domain_name, json_object):
        name = json_object["name"]

        dtype_name = json_object["dtype"]
        if dtype_name in ("int", "int32"):
            dtype = tf.int32
        elif dtype_name in ("float", "float32"):
            dtype = tf.float32
        elif dtype_name == "text":
            dtype = tf.string
        else:
            raise ValueError("UnSupported dtype: {}".format(dtype_name))

        dimension = json_object.get("dimension", 1)
        format_name = json_object.get("format", "text")
        format = FeatureFormat[format_name.upper()]

        transform = cu.ModuleConfig.parse(
            name="transform",
            json_object=json_object.get("transform", {"type": "identity"})
        )

        params = json_object.get("params", dict())

        return FeatureField(
            name=name,
            domain_name=domain_name,
            dimension=dimension,
            dtype=dtype,
            format=format,
            transform=transform,
            params=params
        )

    def __str__(self):
        return "name={},dtype={},format={},group={},transform={}".format(
            self._name,
            self._dtype,
            self._format,
            self._group,
            self._transform.type
        )


class DomainFeature(object):
    def __init__(self, name, max_length, length_field, elements):
        self._name = name
        self._max_length = max_length
        self._length_field = length_field
        self._elements = elements

    def full_name(self, field_name):
        return self._name + _PROPERTY_NAME_SEPARATOR + field_name

    @property
    def name(self):
        return self._name

    @property
    def max_length(self):
        return self._max_length

    @property
    def length_field(self):
        return self._length_field

    @property
    def elements(self):
        return self._elements

    def __iter__(self):
        return iter(self._elements)

    def __getitem__(self, item):
        return self._elements[item]

    @staticmethod
    def parse(domain_name, json_object):
        max_length = int(json_object.get("max_length", 1))
        length_field = json_object.get("length")
        element_feat_conf_list = [
            FeatureField.parse(domain_name, obj)
            for obj in json_object["elements"]
        ]

        return DomainFeature(
            name=domain_name,
            max_length=max_length,
            length_field=length_field,
            elements=element_feat_conf_list
        )

    def __str__(self):
        return "{}[MAX_LEN={}],lenth_filed={}, {} fields".format(
            self._name,
            self._max_length,
            self._length_field,
            len(self._elements)
        )


class InputFeatureConfig(object):
    def __init__(self, user, context, candidate, exposure, ipv):
        self._user_feat_conf = user
        self._context_conf = context
        self._candidate_feat_conf = candidate
        self._exposure_feat_conf = exposure
        self._ipv_feat_conf = ipv

    def __iter__(self):
        return iter(
            [
                f for f in [self._user_feat_conf, self._context_conf, self._candidate_feat_conf,
                        self._exposure_feat_conf, self._ipv_feat_conf]
                if f is not None
            ]
        )

    @property
    def user_feature_conf(self):
        return self._user_feat_conf

    @property
    def context_feature_conf(self):
        return self._context_conf

    @property
    def candidate_feature_conf(self):
        return self._candidate_feat_conf

    @property
    def exposure_feature_conf(self):
        return self._exposure_feat_conf

    @property
    def ipv_feature_conf(self):
        return self._ipv_feat_conf

    @staticmethod
    def parse(json_object):
        def parse_domain(domain_name):
            try:
                return DomainFeature.parse(
                    domain_name,
                    json_object[domain_name]
                )
            except KeyError:
                return None

        return InputFeatureConfig(
            user=parse_domain("user"),
            candidate=parse_domain("candidate"),
            exposure=parse_domain("exposure"),
            ipv=parse_domain("ipv"),
            context=parse_domain("context")
        )
