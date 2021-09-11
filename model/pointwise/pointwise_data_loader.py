# -*- coding: utf-8 -*-
"""
Dataset loader for point-wise ranking model
"""
import tensorflow as tf

from config.feature_config import FeatureField, FeatureFormat, DomainFeature
from loader import loader_base
from loader import parser
from ..model_meta import MetaType, model


class OutputFields:
    """
    Constants used as field names of dataset outputs
    """
    LABEL = "__label__"
    DOMAIN_LENGTH = "__length__"


@model("pointwise", MetaType.DataLoader)
class PointwiseModelDataLoader(loader_base.DataLoader):
    def __init__(self, *args, **kwargs):
        super(PointwiseModelDataLoader, self).__init__(*args, **kwargs)
        
    @staticmethod
    def parse_field(max_length, field_conf, raw_input):
        """
        :type field_conf: FeatureField
        """
        if field_conf.format is FeatureFormat.TEXT:
            return parser.split_and_convert(
                raw_input,
                out_type=field_conf.dtype,
                delimiter=",",
                pad_to=field_conf.dimension * max_length,
                shape=[max_length, field_conf.dimension],
                op_name=field_conf.full_name
            )
        elif field_conf.format is FeatureFormat.BASE64:
            return parser.decode_base64(
                raw_input,
                out_type=field_conf.dtype,
                pad_to=field_conf.dimension * max_length,
                shape=[max_length, field_conf.dimension]
            )
        else:
            # Should never go into here
            assert False

    # TODO: add model label
    def required_fields(self):
        field_names = []
        for domain_features in self._arch_config.feature_config:
            if domain_features.length_field:
                field_names.append(domain_features.length_field)

            field_names.extend(
                f.name for f in domain_features
            )

        field_names.append(self._arch_config.model_config.label_field)
        return field_names

    def _train_data_defaults(self):
        return [""] * (len(self.required_fields()) - 1) + [0]

    def _predict_data_defaults(self):
        return self._train_data_defaults()

    def _train_data_parser(self, *fields):
        outputs = dict()

        # Parse feature fields
        fields_iter = iter(fields)
        for domain_features in self._arch_config.feature_config:  # type: DomainFeature
            if domain_features.length_field:
                outputs[domain_features.full_name(OutputFields.DOMAIN_LENGTH)] = tf.string_to_number(
                    next(fields_iter), tf.int32
                )

            for field_conf, field in zip(domain_features, fields_iter):  # type: (FeatureField, object)
                outputs[field_conf.full_name] = PointwiseModelDataLoader.parse_field(
                    domain_features.max_length,
                    field_conf,
                    field
                )

        label = next(fields_iter)
        outputs[OutputFields.LABEL] = label
        return outputs, label

    def _predict_data_parser(self, *fields):
        return self._train_data_parser(*fields)
