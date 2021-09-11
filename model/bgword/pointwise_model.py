# -*- coding: utf-8 -*-
import logging
from argparse import Namespace

import tensorflow as tf

from util import ModeKeys
from config import FeatureField, DomainFeature, ArchitectureConfig, TrainConfig
from model.pointwise.pointwise_data_loader import OutputFields
from module import encoder, defaults, hooks
from module import projector
from module import checkpoint_loader
from ..model_meta import MetaType, model
from .pointwise_config import PointwiseModelConfig
from . import attention

# Constant variables for reading configurations
_GROUP_PARAM = "group"
_GROUP_ITEM = "item"
_GROUP_BEHAVIOR = "behavior"


@model("bgword", MetaType.ModelBuilder)
class PointwiseModel(object):
    def __init__(self, arch_config, train_config):
        """
        :type train_config: TrainConfig
        :type arch_config: ArchitectureConfig
        """
        self._model_config = arch_config.model_config # type: PointwiseModelConfig
        self._feature_config = arch_config.feature_config
        self._train_config = train_config # type: TrainConfig

    def model_fn(self, features, labels, mode):
        model_output = self._build_model(features)

        if mode is ModeKeys.TRAIN:
            return self._train(model_output, labels)
        elif mode is ModeKeys.EVAL:
            return self._eval(model_output, labels)
        elif mode is ModeKeys.PREDICT:
            return self._predict(features, model_output)


    def _predict(self, features, model_output,mode=tf.estimator.ModeKeys.PREDICT):
        outputs = dict(
            score=model_output.prob,
            label=features[OutputFields.LABEL]
        )
        total_auc, current_auc = tf.metrics.auc(
            labels=tf.cast(features[OutputFields.LABEL], tf.float32),
            predictions=model_output.prob,
            num_thresholds=2000
        )

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=outputs,
            prediction_hooks=[
                tf.train.LoggingTensorHook(
                    {
                        "pre_current_auc": current_auc,
                        "pre_total_auc": total_auc
                    },
                    every_n_iter=1000
                ),
                hooks.GraphPrinterHook()]
        )

    def _train(self, model_output, labels, mode=tf.estimator.ModeKeys.TRAIN):
        assert labels is not None
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(labels, tf.float32),
            logits=model_output.logits
        ))

        total_auc, current_auc = tf.metrics.auc(
            labels=tf.cast(labels, tf.float32),
            predictions=model_output.prob,
            num_thresholds=2000
        )

        max_gradient_norm = None
        if self._train_config.max_gradient_norm > 0:
            max_gradient_norm = self._train_config.max_gradient_norm
        logging.info("Gradient clipping is turned {}".format("ON" if max_gradient_norm else "OFF"))

        trainable_vars = [v for v in tf.trainable_variables(scope = "final_class")]

        logging.info("Trainable variables:")
        for var in trainable_vars:
            print("\t{}:{}".format(var.name,var.shape))

        global_step = tf.train.get_global_step()
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=self._train_config.learning_rate.learning_rate(global_step),
            optimizer="Adam",
            summaries=[
                "loss"
            ],
            variables=trainable_vars,
            clip_gradients=max_gradient_norm
        )

        if self._train_config.init_checkpoint:
            checkpoint_loader.init_from_pretrained_checkpoint(self._train_config.init_checkpoint)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            training_hooks=[
                tf.train.LoggingTensorHook(
                    {
                        "train_loss": loss,
                        "train_total_auc":total_auc,
                        "train_current_auc":current_auc
                    },
                    every_n_iter=200
                ),
                hooks.GraphPrinterHook()
            ]
        )

    def _eval(self, model_output, labels, mode=tf.estimator.ModeKeys.EVAL):
        assert labels is not None
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(labels, tf.float32),
            logits=model_output.logits
        ))

        total_auc, current_auc = tf.metrics.auc(
            labels=tf.cast(labels, tf.float32),
            predictions=model_output.prob,
            num_thresholds=2000
        )

        # if self._train_config.init_checkpoint:
        #     checkpoint_loader.init_from_pretrained_checkpoint(self._train_config.init_checkpoint)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_auc,#loss,
            #predictions={'total_auc':total_auc, "current_auc":current_auc},
            evaluation_hooks=[
                tf.train.LoggingTensorHook(
                    {
                        "eval_loss": loss,
                        "eval_total_auc":total_auc,
                        "eval_current_auc":current_auc
                    },
                    every_n_iter=500
                ),
                hooks.GraphPrinterHook()
            ]
        )

    def _build_model(self, features):
        # Encode inputs to dense embedding
        input_embeddings = self._encode_input_as_embedding(features)

        # Encode sequential embeddings
        #exp_seq_item_features = self._encode_sequence(
        #    self._feature_config.exposure_feature_conf,
        #    input_embeddings,
        #    scope_name="exposure_seq_encoder"
        #)

        ipv_seq_item_features = self._encode_sequence(
            self._feature_config.ipv_feature_conf,
            input_embeddings,
            scope_name="ipv_seq_encoder"
        )

        # Encode candidate feature, shape: B*D
        candidate_feature = self._encode_candidate_features(input_embeddings)

        # Perform attention
        #exp_cross_features = self._perform_attention(
        #    candidate_feature,
        #    item_features=exp_seq_item_features,
        #    seq_length=features[self._feature_config.exposure_feature_conf.full_name(OutputFields.DOMAIN_LENGTH)],
        #    hidden_size=self._model_config.attention_hidden_size,
        #    scope_name="exposure_attention"
        #)

        ipv_cross_features = self._perform_attention(
            candidate_feature,
            item_features=ipv_seq_item_features,
            seq_length=features[self._feature_config.ipv_feature_conf.full_name(OutputFields.DOMAIN_LENGTH)],
            hidden_size=self._model_config.attention_hidden_size,
            scope_name="ipv_attention"
        )

        # Encode user features
        user_features = self._encode_user_features(input_embeddings)

        final_features = tf.concat(
            #[candidate_feature, ipv_cross_features, exp_cross_features, user_features],
            [candidate_feature, ipv_cross_features, user_features],
            axis=1
        )

        with defaults.variable_scope("final_class"):
            # Final classifier
            final_features = projector.multi_layer_perceptron(
                final_features,
                layers=self._model_config.classifier_projection,
                activation_fn=tf.nn.tanh,
                scope_name="classifier"
            )

            logits = tf.layers.dense(
                final_features, units=1, activation=None,
                kernel_initializer=defaults.weight_initializer()
            )
            logits = tf.squeeze(logits, axis=1)
            prob = tf.sigmoid(logits)

        return Namespace(
            logits=logits,
            prob=prob
        )

    def _encode_input_as_embedding(self, features):
        input_embeddings = dict()
        context = encoder.EncoderContext()
        for domain_features in self._feature_config:
            for field_conf in domain_features:  # type: FeatureField
                input_embeddings[field_conf.full_name] = encoder.encode_input(
                    features[field_conf.full_name],
                    field_conf,
                    scope_name=domain_features.name,
                    context=context
                )
        return input_embeddings

    def _encode_sequence(self, domain_features, input_features, scope_name):
        """
        Encode a sequence of behavior items and generate contextual item embeddings
        :type domain_features: DomainFeature
        """
        with defaults.variable_scope(scope_name):
            item_features_list = [input_features[feature_field.full_name] for feature_field in domain_features]
            item_features = tf.concat(item_features_list, axis=2)

            for i in range(self._model_config.sequence_encoder_blocks):
                with tf.variable_scope("block_{}".format(i)):
                    with tf.variable_scope("item_encoder"):
                        item_features = self._do_encode_sequence_with_dnn(
                            item_features,
                            hidden_size=self._model_config.item_encoder_hidden_size,
                            max_length=domain_features.max_length
                        )

            return item_features

    @staticmethod
    def _do_encode_sequence_with_dnn(input_features, hidden_size, max_length):
        return projector.multi_layer_perceptron(
            input_features,
            activation_fn=tf.nn.tanh,
            layers=[hidden_size],
            scope_name="projection"
        )

    def _encode_candidate_features(self, input_embeddings):
        # B*1*D
        candidate_features = tf.concat([
            input_embeddings[field.full_name]
            for field in self._feature_config.candidate_feature_conf
        ], axis=2)

        # Remove the second dummy dimension
        # Now converts to B*D
        candidate_features = tf.squeeze(candidate_features, axis=1)

        # Do projection
        scope_name = "candidate_encoder"
        candidate_features = projector.multi_layer_perceptron(
            candidate_features,
            layers=self._model_config.candidate_encoder_projection[:-1],
            activation_fn=tf.nn.tanh,
            scope_name="candidate_encoder"
        )

        with defaults.variable_scope(scope_name):
            return tf.layers.dense(
                inputs=candidate_features,
                units=self._model_config.candidate_encoder_projection[-1],
                activation=None,
                kernel_initializer=defaults.weight_initializer(),
                name="candidate_feature_head"
            )

    def _perform_attention(self, candidate_features, item_features, seq_length, hidden_size, scope_name):
        """
        Compute cross feature between candidate and sequential features via attention mechanism
        """
        with defaults.variable_scope(scope_name):
            attender = attention.AttentionLayerBahdanau()
            return attender._build(
                query=candidate_features,
                keys=item_features,
                values=item_features,
                values_length=seq_length,
                num_units=hidden_size
            )

    def _encode_user_features(self, input_embeddings):
        # B*1*D
        user_features = tf.concat([
            input_embeddings[field.full_name]
            for field in self._feature_config.user_feature_conf
        ], axis=2)

        # Remove the second dummy dimension
        # Now converts to B*D
        return tf.squeeze(user_features, axis=1)
