# -*- coding: utf-8 -*-
"""
Environmental setup
"""
import logging
import os
import sys

import tensorflow as tf


def setup_logging():
    logging_formatter = logging.Formatter("%(asctime)s-%(levelname)s-%(message)s")

    # Setup tensorflow logger
    tf.logging.set_verbosity(tf.logging.INFO)
    logger = logging.getLogger('tensorflow')
    logger.propagate = False
    logger.handlers[0].formatter = logging_formatter

    # Setup common logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging_formatter)
    root.addHandler(handler)


def setup_path():
    parent_path = os.path.abspath(os.path.dirname(__file__) + "/..")
    sys.path.insert(0, parent_path)


def set_dist_env(dist_params):
    worker_hosts = dist_params['worker_hosts'].split(',')
    worker_hosts = worker_hosts[1:]  # the rest as worker
    task_index = dist_params['task_index']
    job_name = dist_params['job_name']

    if job_name == "worker" and task_index == 0:
        job_name = "chief"
    # the others as worker
    if job_name == "worker" and task_index > 0:
        task_index -= 1
    del os.environ['TF_CONFIG']

    slice_count = len(worker_hosts) + 1
    if job_name == "ps":
        return 1, 0
    else:
        slice_id = int(dist_params['task_index'])
        return slice_count, slice_id
