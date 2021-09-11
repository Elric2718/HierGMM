# -*- coding: utf-8 -*-
"""
Train the model on PAI-TensorFlow
"""
from __future__ import absolute_import
import argparse
import logging

import pandas as pd
import tensorflow as tf
import sys
sys.path.append("../../")

import model
import numpy as np
from model import hcd
import loader
from config import TrainConfig
from config import ArchitectureConfig
from config import LearningRate
from util import env
from util import path
from util import args_processing as ap
from util import ModeKeys

import time
from util import logz

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, help="CSV Training Data file path")
    parser.add_argument("--test_file", type=str, help="CSV Testing Data file path")
    parser.add_argument("--output_path", type=str, help="CSV Output path")

    parser.add_argument("--max_steps", type=int, help="Number of iterations before stopping")
    parser.add_argument("--multiplier", type=int, help="Multiplier for max steps. A dummy variable.", default=1)
    parser.add_argument("--snapshot", type=int, help="Number of iterations to dump model")
    parser.add_argument("--checkpoint_dir", type=str, help="Path of the checkpoint path")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--model", type=str, help="model type")
    parser.add_argument("--algo", type=str, help="hier")
    parser.add_argument("--infer_strategy", type=str, help="Inference Strategy")

    parser.add_argument("--max_gradient_norm", type=float, default=0.)

    # If both of the two options are set, `model_config` is preferred
    parser.add_argument("--arch_config_path", type=str, default=None, help="Path of model configs")
    parser.add_argument("--arch_config", type=str, default=None, help="base64-encoded model configs")

    # init checkpoint
    parser.add_argument("--buckets", type=str, default='')
    parser.add_argument("--init_checkpoint", type=str, default='')
    parser.add_argument("--init_step", type=int, default=2000)
    parser.add_argument("--init_steps_val", type=str, default="2000")

    # HCD configuration
    parser.add_argument("--n_groups", type=int, default=2)
    parser.add_argument("--n_sampling", type=int, default=5)
    parser.add_argument("--EM_rounds", type=int, default=5)
    parser.add_argument("--EM_unsup_warningup", type=int, default=5)

    parser.add_argument("--unsup_temp", type=float, default=1.)

    return parser.parse_known_args()[0]

def setup_logger(logdir, args):
    # Configure output director for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters 
    params = {key: val for key, val in args.__dict__.items()}   
    logz.save_params(params)

def main():
    args = parse_args()
    logging.info("Main arguments:")
    for k, v in args.__dict__.items():
        print("{}={}".format(k, v))

    logdir = f'log/{args.algo}-{str(time.time())}.log'
    setup_logger(logdir, args)

    # Check if the specified path has a existed model
    full_checkpoint_dir = args.checkpoint_dir
    #if tf.gfile.Exists(path.join_oss_path(full_checkpoint_dir, path.CHECKPOINT_FILE_NAME)):
    #    raise ValueError("Model %s has already existed, please delete them and retry" % args.checkpoint_dir)


    #model_meta = model.get_model_meta(args.model)  # type: model.ModelMeta

    # Load architecture configuration
    #arch_conf = ap.parse_arch_config_from_args(model_meta, args)  # type: ArchitectureConfig

    # Figure out model initialization
    init_checkpoint_file = ap.get_init_checkpoint_file(args.buckets, args.init_checkpoint, args.init_step) if args.init_step > 0 else None
    # improvable: store assignemnts, then deattach the global models from the hier/clus2pred models.
    global_checkpoint_files = {}
    for global_step in args.init_steps_val.split(','):
         global_checkpoint_files[f"Global-{int(global_step)}"] = ap.get_init_checkpoint_file(args.buckets, args.init_checkpoint, int(global_step)) if int(global_step) > 0 else None

    hcd_model = hcd.HierCloudDevice(args = args,\
                            algo = args.algo,\
                            n_groups = args.n_groups,\
                            config = None,\
                            init_checkpoint = init_checkpoint_file,\
                            global_checkpoints = global_checkpoint_files,\
                            log_every=100,\
                            continue_training=True,\
                            store_attributes=True,\
                            logz = logz)

    hcd_model._train_and_eval(args.train_file, args.test_file, args.infer_strategy)

    """
    hcd_model._train(args.train_file)
    

    assignments = hcd_model._infer_cluster(args.test_file, args.infer_strategy)    
    if assignments is None:
        assert args.n_groups == 1
    logz.log_tabular("Assignment", ','.join([str(g_id) + ":" + str(np.sum(assignments == g_id)) for g_id in range(args.n_groups)]))
    logz.dump_tabular()
    output = {'size': ','.join([str(g_id) + ":" + str(np.sum(assignments == g_id)) if assignments is not None else f"0:{sum([1 for _ in open(args.test_file)])}" for g_id in range(args.n_groups)])}

    results= {}
    for group_id in range(args.n_groups):
        results[group_id] = hcd_model._eval(args.test_file, assignments, group_id, group_id)
    output[args.algo].append(','.join([str(key) + ":" + str(round(val['loss'], 4)) for key, val in results.items()]))
        

    pd.DataFrame(output).to_csv(args.output_path + "_" + args.algo + ".csv", sep = ",")
    """


if __name__ == '__main__':

    env.setup_logging()
    env.setup_path()
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
