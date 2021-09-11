# -*- coding: utf-8 -*-#
import tensorflow as tf


def get_odps_writer(table_name, slice_id):
    return tf.python_io.TableWriter(table_name, slice_id=slice_id)
