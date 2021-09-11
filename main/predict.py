# -*- coding: utf-8 -*-
"""
Inference user embeddings on PAI-TensorFlow
"""
import logging
import sys
import time
import argparse

import tensorflow as tf
from tensorflow.python.platform import gfile

import loader
import model
from util import ModeKeys
from util import dumper
from util import env
from util import path
from util import args_processing as ap


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=str, help="Kernels configuration for CNN")
    parser.add_argument("--outputs", type=str, help="Destination of the table ")
    parser.add_argument("--buckets", type=str, help="Worker task index")
    parser.add_argument("--checkpoint_dir", type=str, help="Path of the checkpoint path")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--prefetch", type=int, default=40000)

    # Arguments for distributed inference
    parser.add_argument("--task_index", type=int, help="Task index")
    parser.add_argument("--ps_hosts", type=str, help="")
    parser.add_argument("--worker_hosts", type=str, help="")
    parser.add_argument("--job_name", type=str)

    return parser.parse_known_args()[0]


def _do_prediction(result_iter, writer, args):
    logging.info("Start inference......")
    t_start = t_batch_start = time.time()
    report_gap = 10000

    indices = [0, 1]
    record_buffer = []
    for i, prediction in enumerate(result_iter, 1):
        record_buffer.append([
            float(prediction["label"]),
            float(prediction["score"])
        ])

        if len(record_buffer) >= 256:
            writer.write(record_buffer, indices)
            record_buffer = []

        if i % report_gap == 0:
            t_now = time.time()
            logging.info("[{}]Processed {} samples, {} records/s, cost {} s totally, {} records/s averagely".format(
                args.task_index,
                i,
                report_gap / (t_now - t_batch_start),
                (t_now - t_start),
                i / (t_now - t_start)
            ))
            t_batch_start = t_now

    if len(record_buffer) > 0:
        writer.write(record_buffer, indices)

    writer.close()


def main():
    # Parse arguments and print them
    args = parse_args()
    print("\nMain arguments:")
    for k, v in args.__dict__.items():
        print("{}={}".format(k, v))

    # Setup distributed inference
    dist_params = {
        "task_index": args.task_index,
        "ps_hosts": args.ps_hosts,
        "worker_hosts": args.worker_hosts,
        "job_name": args.job_name
    }
    slice_count, slice_id = env.set_dist_env(dist_params)

    # Load model configs
    full_checkpoint_dir = path.join_oss_path(args.buckets, args.checkpoint_dir)
    train_args = ap.load_train_arguments(full_checkpoint_dir)

    model_meta = model.get_model_meta(train_args.model)  # type: model.ModelMeta
    arch_conf = ap.load_arch_config(model_meta, full_checkpoint_dir)
    print(arch_conf)

    rank_model = model_meta.model_builder(
        arch_config=arch_conf,
        train_config=None,
    )

    estimator = tf.estimator.Estimator(
        model_fn=rank_model.model_fn,
        model_dir=args.checkpoint_dir,
        config=tf.estimator.RunConfig(
            session_config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(allow_growth=False),
                allow_soft_placement=True
            ),
            save_checkpoints_steps=10000,
            keep_checkpoint_max=1
        )
    )

    checkpoint_path = None
    if args.step > 0:
        checkpoint_path = full_checkpoint_dir + "/model.ckpt-{}".format(args.step)
        if not gfile.Exists(checkpoint_path + ".index"):
            checkpoint_path = full_checkpoint_dir + "/model.ckpt-{}".format(args.step+1)

    result_iter = estimator.predict(
        model_meta.data_loader_builder(
            arch_config=arch_conf,
            mode=ModeKeys.PREDICT,
            source=loader.ODPSDataSource(
                table_name=args.tables,
                slice_id=slice_id,
                slice_count=slice_count
            ),
            shuffle=0,
            batch_size=args.batch_size,
            prefetch=args.prefetch,
            parallel_calls=4,
            repeat=1,
        ).input_fn,
        checkpoint_path=checkpoint_path
    )

    odps_writer = dumper.get_odps_writer(
        args.outputs,
        slice_id=slice_id
    )
    _do_prediction(result_iter, odps_writer, args)

if __name__ == '__main__':
    env.setup_path()
    env.setup_logging()
    main()
