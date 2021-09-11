# -*- coding: utf-8 -*-
"""
Print required fields used for generating select SQL
"""
from __future__ import absolute_import
import argparse

import model
import loader
from config import ArchitectureConfig
from util import args_processing as ap
from util import ModeKeys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model type")
    # If both of the two options are set, `model_config` is preferred
    parser.add_argument("--arch_config_path", type=str, default=None, help="Path of model configs")
    parser.add_argument("--arch_config", type=str, default=None, help="base64-encoded model configs")

    return parser.parse_known_args()[0]


def main():
    args = parse_args()
    model_meta = model.get_model_meta(args.model)  # type: model.ModelMeta

    # Load architecture configuration
    arch_conf = ap.parse_arch_config_from_args(model_meta, args)  # type: ArchitectureConfig

    data_loader = model_meta.data_loader_builder(
        arch_config=arch_conf,
        mode=ModeKeys.TRAIN,
        source=None,
        shuffle=1000,
        batch_size=None,
        prefetch=10000,
        parallel_calls=4,
        repeat=None
    )  # type: loader.DataLoader

    fields = data_loader.required_fields()
    print("""
SELECT
\t{}
FROM 
    """.format("\n\t,".join(fields))
          )


if __name__ == '__main__':
    main()
