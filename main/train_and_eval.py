# -*- coding: utf-8 -*-
"""
Train the model on PAI-TensorFlow
"""
from __future__ import absolute_import
import argparse
import logging

import tensorflow as tf
from tensorflow.contrib.distribute.python import cross_tower_ops as cross_tower_ops_lib

import model
import loader
from config import TrainConfig, LearningRate
from config import ArchitectureConfig
from util import env
from util import path
from util import args_processing as ap
from util import ModeKeys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=str, help="Kernels configuration for CNN")
    parser.add_argument("--buckets", type=str, help="Worker task index")

    parser.add_argument("--max_steps", type=int, help="Number of iterations before stopping")
    parser.add_argument("--snapshot", type=int, help="Number of iterations to dump model")
    parser.add_argument("--checkpoint_dir", type=str, help="Path of the checkpoint path")
    parser.add_argument("--learning_rate", type=str, default=0.001)
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--model", type=str, help="model type")
    parser.add_argument("--init_checkpoint", type=str, default="", help="Path of the checkpoint path")
    parser.add_argument("--init_step", type=int, default=0, help="Path of the checkpoint path")

    parser.add_argument("--max_gradient_norm", type=float, default=0.)

    # If gpu is set to more than 100, MirrorStrategy is used
    parser.add_argument("--gpu", type=int, default=100)

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
    full_checkpoint_dir = path.join_oss_path(args.buckets, args.checkpoint_dir)

    if tf.gfile.Exists(path.join_oss_path(full_checkpoint_dir, path.CHECKPOINT_FILE_NAME)):
        raise ValueError("Model %s has already existed, please delete them and retry" % args.checkpoint_dir)

    model_meta = model.get_model_meta(args.model)  # type: model.ModelMeta

    # Load architecture configuration
    arch_conf = ap.parse_arch_config_from_args(model_meta, args)  # type: ArchitectureConfig

    # Dump arguments and model architecture configuration to OSS
    ap.dump_train_arguments(full_checkpoint_dir, args)
    ap.dump_arch_config(full_checkpoint_dir, arch_conf)

    # Figure out model initialization
    init_checkpoint_file = ap.get_init_checkpoint_file(args.buckets, args.init_checkpoint, args.init_step)

    # get train and test tabel
    tables=args.tables.split(",")
    if len(tables)==1:
        train_table_name = tables[0]
        test_table_name = tables[0]
        logging.info("test tabel = train tabel")
    elif  len(tables)==2:
        train_table_name=tables[0]
        test_table_name=tables[1]
    else:
        raise ValueError("Tabels %s has error input" % args.tabels)

    # Build model
    rank_model = model_meta.model_builder(
        arch_config=arch_conf,
        train_config=TrainConfig(
            learning_rate=LearningRate(args.learning_rate),
            batch_size=args.batch_size,
            max_gradient_norm=args.max_gradient_norm,
            log_every=100,
            init_checkpoint=init_checkpoint_file
        )
    )

    if args.gpu > 100:
        cross_tower_ops = cross_tower_ops_lib.AllReduceCrossTowerOps(
            'nccl'
        )
        assert args.gpu % 100 == 0
        distribution = tf.contrib.distribute.MirroredStrategy(
            num_gpus=args.gpu / 100, cross_tower_ops=cross_tower_ops,
            all_dense=False
        )
    else:
        distribution = None

    estimator = tf.estimator.Estimator(
        model_fn=rank_model.model_fn,
        model_dir=path.join_oss_path(args.buckets, args.checkpoint_dir),
        config=tf.estimator.RunConfig(
            session_config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(allow_growth=False),
                allow_soft_placement=True,
            ),
            save_checkpoints_steps=args.snapshot,
            keep_checkpoint_max=40,
            train_distribute=distribution
        )
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=model_meta.data_loader_builder(
            arch_config=arch_conf,
            mode=ModeKeys.TRAIN,
            source=loader.ODPSDataSource(table_name=train_table_name),
            shuffle=1000,
            batch_size=args.batch_size,
            prefetch=10000,
            parallel_calls=4,
            repeat=1
        ).input_fn,
        max_steps=1000000)

    eval_spec = tf.estimator.EvalSpec(
        input_fn=model_meta.data_loader_builder(
            arch_config=arch_conf,
            mode=ModeKeys.TRAIN,
            source=loader.ODPSDataSource(table_name=test_table_name),
            shuffle=1000,
            batch_size=args.batch_size,
            prefetch=10000,
            parallel_calls=4,
            repeat=1
        ).input_fn,
        steps=None,
        start_delay_secs=5000,
        throttle_secs=600)

    tf.estimator.train_and_evaluate(
        estimator,
        train_spec,
        eval_spec
    )


if __name__ == '__main__':
    env.setup_logging()
    env.setup_path()
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
