# -*- coding: utf-8 -*-
"""
Train the model on PAI-TensorFlow
"""
from __future__ import absolute_import
import argparse
import logging

import tensorflow as tf
import sys
sys.path.append("../../")

import model
import loader
from config import TrainConfig
from config import ArchitectureConfig
from config import LearningRate
from util import env
from util import path
from util import args_processing as ap
from util import ModeKeys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, help="CSV Data file path")

    parser.add_argument("--max_steps", type=int, help="Number of iterations before stopping")
    parser.add_argument("--snapshot", type=int, help="Number of iterations to dump model")
    parser.add_argument("--checkpoint_dir", type=str, help="Path of the checkpoint path")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--model", type=str, help="model type")

    parser.add_argument("--max_gradient_norm", type=float, default=0.)

    # If both of the two options are set, `model_config` is preferred
    parser.add_argument("--arch_config_path", type=str, default=None, help="Path of model configs")
    parser.add_argument("--arch_config", type=str, default=None, help="base64-encoded model configs")

    return parser.parse_known_args()[0]


def main():
    args = parse_args()
    logging.info("Main arguments:")
    for k, v in args.__dict__.items():
        print("{}={}".format(k, v))

    # Check if the specified path has a existed model
    full_checkpoint_dir = args.checkpoint_dir
    #if tf.gfile.Exists(path.join_oss_path(full_checkpoint_dir, path.CHECKPOINT_FILE_NAME)):
    #    raise ValueError("Model %s has already existed, please delete them and retry" % args.checkpoint_dir)


    model_meta = model.get_model_meta(args.model)  # type: model.ModelMeta

    # Load architecture configuration
    arch_conf = ap.parse_arch_config_from_args(model_meta, args)  # type: ArchitectureConfig

    # Build model
    rank_model = model_meta.model_builder(
        arch_config=arch_conf,
        train_config=TrainConfig(
            learning_rate=LearningRate(args.learning_rate),
            batch_size=args.batch_size,
            max_gradient_norm=args.max_gradient_norm,
            log_every=100,
            init_checkpoint=None
        )
    )

    estimator = tf.estimator.Estimator(
        model_fn=rank_model.model_fn,
        model_dir=args.checkpoint_dir,
        config=tf.estimator.RunConfig(
            session_config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(allow_growth=False),
                allow_soft_placement=True,
            ),
            save_checkpoints_steps=args.snapshot,
            keep_checkpoint_max=40,
            train_distribute=None
        )
    )

    estimator.train(
        model_meta.data_loader_builder(
            arch_config=arch_conf,
            mode=ModeKeys.TRAIN,
            source=loader.CsvDataSource(file_path=args.data_file, delimiter=";"),
            shuffle=10,
            batch_size=args.batch_size,
            prefetch=100,
            parallel_calls=4,
            repeat=None
        ).input_fn,
        steps=args.max_steps
    )


if __name__ == '__main__':
    env.setup_logging()
    env.setup_path()
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
