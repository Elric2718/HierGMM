# -*- coding: utf-8 -*-
from ..model_meta import MetaType, model


class PointwiseModelConfig(object):
    def __init__(self):
        # initialize and give proper type hints
        self.sequence_encoder = 0
        self.sequence_encoder_blocks = 0
        self.item_encoder_hidden_size = 0
        self.behavior_encoder_hidden_size = 0
        self.attention_hidden_size = 0
        self.candidate_encoder_projection = []
        self.classifier_projection = []
        self.label_field = ""
        self.all_fields = []

    @staticmethod
    @model("global_bgword", MetaType.ConfigParser)
    def parse(json_obj):
        conf = PointwiseModelConfig()
        conf.sequence_encoder = str(json_obj.get("sequence_encoder"))
        conf.sequence_encoder_blocks = int(json_obj.get('sequence_encoder_blocks'))
        conf.item_encoder_hidden_size = int(json_obj.get('item_encoder_hidden_size'))
        conf.behavior_encoder_hidden_size = int(json_obj.get("behavior_encoder_hidden_size"))
        conf.attention_hidden_size = int(json_obj.get("attention_hidden_size"))
        conf.classifier_projection = list(int(v) for v in json_obj.get("classifier_projection"))
        conf.candidate_encoder_projection = list(int(v) for v in json_obj.get("candidate_encoder_projection"))
        conf.label_field = str(json_obj.get("label_field"))
        conf.all_fields = json_obj.get("all_fields")
        conf.fields_category = json_obj.get("fields_category")

        return conf
