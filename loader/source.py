# -*- coding: utf-8 -*-
"""
Encapsule data sources.

All data source classes are stateless. They only hold the data source opening and splitting logics.
The returned dataset must have the fields well separated.
Since the field defaults depend on the dataset format, we leave it in the concrete dataset classes
"""

import tensorflow as tf


class ODPSDataSource(object):
    """
    The data source is from a MaxCompute table
    """
    def __init__(self, table_name, slice_id=0, slice_count=1):
        self._table_name = table_name
        self._slice_id = slice_id
        self._slice_count = slice_count

        assert 0 <= self._slice_id < self._slice_count, "Invalid slice_id"

    @property
    def table_name(self):
        return self._table_name

    def open(self, defaults, field_names):
        """
        Open dataset
        :param defaults: Default values for each column.
        :return: Opened dataset object
        """
        selected_cols = None if not field_names else ",".join(field_names)
        return tf.data.TableRecordDataset(
            self._table_name,
            record_defaults=defaults,
            slice_id=self._slice_id,
            slice_count=self._slice_count,
            selected_cols=selected_cols
        )


class CsvDataSource(object):
    """
    The data source is from a CSV text file from local disk
    It's mainly used for debugging in local IDE
    """
    def __init__(self, file_path, delimiter="\t", header=False, na_value='', use_quote_delim=False):
        self._file_path = file_path
        self._delimiter = delimiter
        self._header = header
        self._na_value = na_value
        self._use_quote_delim = use_quote_delim

    @property
    def delimiter(self):
        return self._delimiter

    @property
    def file_path(self):
        return self._file_path

    def open(self, defaults, field_names):
        return tf.data.experimental.CsvDataset(
            self._file_path,
            record_defaults=defaults,
            field_delim=self._delimiter,
            use_quote_delim=self._use_quote_delim,
            header=self._header,
            na_value=self._na_value,
            select_cols=field_names
        )
