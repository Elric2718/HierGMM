# -*- coding: utf-8 -*-
"""
Argument parsing and dumpling
"""
import copy
import json
import base64
from argparse import Namespace

import tensorflow as tf
from tensorflow.python.platform import gfile

from . import path
import config


_ARCH_CONFIG_FILE_NAME = "arch_conf.json"
_TRAIN_ARGS_FILE_NAME = "train_args.json"


def get_init_checkpoint_file(buckets, checkpoint_file, init_step):
    if checkpoint_file and init_step and buckets:
        full_init_checkpoint_dir = path.join_oss_path(buckets, checkpoint_file)
        full_init_checkpoint_file = full_init_checkpoint_dir + "/model.ckpt-{}".format(init_step)
        if not gfile.Exists(full_init_checkpoint_file + ".index"):
            full_init_checkpoint_file = full_init_checkpoint_dir + "/model.ckpt-{}".format(init_step + 1)

        return full_init_checkpoint_file
    else:
        return None


def parse_arch_config_from_args(model_meta, args):
    """
    Read or parse arch config
    :param model_meta:
    :param args:
    :return:
    """
    if args.arch_config is not None:
        raw_arch_config = json.loads(base64.b64decode(args.arch_config))
    elif args.arch_config_path is not None:
        with tf.gfile.GFile(args.arch_config_path, "r") as fin:
            raw_arch_config = json.load(fin)
    else:
        raise KeyError("Model configuration not found")

    return config.ArchitectureConfig.parse(raw_arch_config, model_meta.arch_config_parser)


def load_arch_config(model_meta, checkpoint_dir):
    """
    Load arch config from OSS
    :param model_meta:
    :param args:
    :return:
    """
    with tf.gfile.GFile(path.join_oss_path(checkpoint_dir, _ARCH_CONFIG_FILE_NAME), "r") as fin:
        return config.ArchitectureConfig.parse(json.load(fin), model_meta.arch_config_parser)


def dump_arch_config(checkpoint_dir, model_arch):
    """
    Dump model configurations to OSS
    :param args: Namespace object, parsed from command-line arguments
    :param model_arch:
    :return:
    """
    with tf.gfile.GFile(
            path.join_oss_path(checkpoint_dir, _ARCH_CONFIG_FILE_NAME),
            "w"
    ) as fout:
        json.dump(model_arch.raw_data, fout)


def dump_train_arguments(checkpoint_dir, args):
    args_dict = copy.copy(args.__dict__)
    args_dict.pop("arch_config")

    with tf.gfile.GFile(
        path.join_oss_path(checkpoint_dir, _TRAIN_ARGS_FILE_NAME),
        "w"
    ) as fout:
        json.dump(args_dict, fout)


def load_train_arguments(checkpoint_dir):
    ns = Namespace()
    with tf.gfile.GFile(
        path.join_oss_path(checkpoint_dir, _TRAIN_ARGS_FILE_NAME),
        "r"
    ) as fin:
        for key, value in json.load(fin).iteritems():
            setattr(ns, key, value)

    return ns
