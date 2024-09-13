# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from SGP (63d33cc)
#   (https://github.com/theNded/SGP/tree/63d33cc8bffde53676d9c4800f4b11804b53b360)
# Copyright (c) 2021 Wei Dong and Heng Yang, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import sys, os

file_path = os.path.abspath(__file__)
p3d_path = os.path.dirname(file_path)
project_path = os.path.dirname(p3d_path)
sys.path.append(project_path)

fcgf_path = os.path.join(project_path, 'ext', 'FCGF')
sys.path.append(fcgf_path)

from tqdm import tqdm

import torch
import numpy as np
import open3d as o3d

from sgp_base import SGPBase
from dataset.threedmatch_train import Dataset3DMatchTrain
from dataset.threedmatch_sgp import Dataset3DMatchSGP
from perception3d.adaptor import DatasetFCGFAdaptor, FCGFConfigParser, load_fcgf_model, fcgf_train, reload_config, register


def verify_config(config):
    assert len(config.overlap_thr) == len(config.max_epoch_per_iter)

    if config.meta_iters is None:
        config.meta_iters = len(config.max_epoch_per_iter)
    else:
        assert len(config.max_epoch_per_iter) == config.meta_iters

    # Shortcuts
    if config.no_verifier:
        config.overlap_thr = [0]*config.meta_iters

    if config.n_epochs_per_new_label is not None:
        assert config.n_epochs_per_new_label > 0, "n_epochs_per_new_label must be positive"
        assert all([max_epoch % config.n_epochs_per_new_label == 0 for max_epoch in config.max_epoch_per_iter]), \
            "all elems in max_epoch_per_iter must be divisible by n_epochs_per_new_label"
        assert config.finetune, "finetune must be enabled for n_epochs_per_new_label"
        
        config.overlap_thr = sum([[thr]*(max_epoch//config.n_epochs_per_new_label) for thr, max_epoch in zip(config.overlap_thr, config.max_epoch_per_iter)], [])
        config.meta_iters = sum(config.max_epoch_per_iter)//config.n_epochs_per_new_label
        config.max_epoch_per_iter = [config.n_epochs_per_new_label]*config.meta_iters

    
    assert not config.self_training, "Disable self-training for SGP."

    return config


class SGP3DRegistration(SGPBase):
    def __init__(self):
        super(SGP3DRegistration, self).__init__()

    # override
    def perception_bootstrap(self, src_data, dst_data, src_info, dst_info,
                             config):
        T, _, fitness, _ = register(src_data, dst_data, 'FPFH', 'RANSAC', None,
                              config)
        if config.debug:
            src_data.paint_uniform_color([1, 0, 0])
            dst_data.paint_uniform_color([0, 1, 0])
            o3d.visualization.draw([src_data.transform(T), dst_data])
        return T, fitness

    # override
    def perception(self, src_data, dst_data, src_info, dst_info, model,
                   config):
        T, _, fitness, _ = register(src_data, dst_data, 'FCGF', 'RANSAC', model,
                              config)
        if config.debug:
            src_data.paint_uniform_color([1, 0, 0])
            dst_data.paint_uniform_color([0, 1, 0])
            o3d.visualization.draw([src_data.transform(T), dst_data])
        return T, fitness

    # override
    def train_adaptor(self, sgp_dataset, config, val_dataset=None):
        fcgf_train(sgp_dataset, val_dataset, config)

    def run(self, config):
        config = verify_config(config)

        config.max_epoch = config.max_epoch_per_iter[0]
        base_pseudo_label_dir = config.pseudo_label_dir
        base_outdir = config.out_dir

        # Bootstrap
        if config.restart_meta_iter < 0:
            pseudo_label_path_bs = os.path.join(base_pseudo_label_dir, 'bs')
            teach_dataset = Dataset3DMatchSGP(config.dataset_path,
                                              config.scenes,
                                              pseudo_label_path_bs, 'teaching',
                                              overlap_thr=0.3) # Conceptually, overlap_thr must be 0.3 for teaching, 
                                                               # since only samples with 30% overlap are used during training.
                                                               # Should not be confused with overlap_thr of SGP verifier.
                                                               # Bug in original SGP code?
            # We need mutual filter for less reliable FPFH
            config.mutual_filter = True
            sgp.teach_bootstrap(teach_dataset, config)

            learn_dataset = DatasetFCGFAdaptor(
                Dataset3DMatchSGP(config.dataset_path, config.scenes,
                                  pseudo_label_path_bs, 'learning',
                                  config.overlap_thr[0]), config)

            if config.test_valid:
                teach_val_dataset = Dataset3DMatchSGP(config.dataset_path,
                                                      config.val_scenes,
                                                      pseudo_label_path_bs, 'teaching',
                                                      overlap_thr=0.3)
                sgp.teach_bootstrap(teach_val_dataset, config)
                learn_val_dataset = DatasetFCGFAdaptor(
                    Dataset3DMatchSGP(config.dataset_path, config.val_scenes,
                                    pseudo_label_path_bs, 'learning',
                                    overlap_thr=0.3), config)
            else:
                learn_val_dataset = None
            config.out_dir = os.path.join(base_outdir, 'bs')
            sgp.learn(learn_dataset, config, learn_val_dataset)

        # Loop
        start_meta_iter = max(config.restart_meta_iter, 0)
        for i in range(start_meta_iter, config.meta_iters-1): # reduced by 1, because one learn step already performed outside the loop
            pseudo_label_path_i = os.path.join(config.pseudo_label_dir,
                                               '{:02d}'.format(i))
            teach_dataset = Dataset3DMatchSGP(config.dataset_path,
                                              config.scenes,
                                              pseudo_label_path_i, 'teaching',
                                              overlap_thr=0.3) # Conceptually, overlap_thr must be 0.3 for teaching, 
                                                               # since only samples with 30% gt overlap are used during training.
                                                               # Should not be confused with overlap_thr of SGP verifier.
                                                               # Bug in original SGP code?

            # No mutual filter results in better FCGF teaching
            config.mutual_filter = False
            model = load_fcgf_model(config)
            sgp.teach(teach_dataset, model, config)

            learn_dataset = DatasetFCGFAdaptor(
                Dataset3DMatchSGP(config.dataset_path, config.scenes,
                                  pseudo_label_path_i, 'learning',
                                  config.overlap_thr[i+1]), config)
            if config.test_valid:
                teach_val_dataset = Dataset3DMatchSGP(config.dataset_path,
                                                      config.val_scenes,
                                                      pseudo_label_path_i, 'teaching',
                                                      overlap_thr=0.3)
                sgp.teach(teach_val_dataset, model, config)
                learn_val_dataset = DatasetFCGFAdaptor(
                                    Dataset3DMatchSGP(config.dataset_path, config.val_scenes,
                                    pseudo_label_path_i, 'learning',
                                    0.3), config)
            else:
                learn_val_dataset = None

            if config.finetune:
                config.resume_dir = config.out_dir
                config = reload_config(config)
                config.max_epoch += config.max_epoch_per_iter[i+1]
            else:
                config.max_epoch = config.max_epoch_per_iter[i+1]

            config.out_dir = os.path.join(base_outdir, '{:02d}'.format(i))

            sgp.learn(learn_dataset, config, learn_val_dataset)


if __name__ == '__main__':
    parser = FCGFConfigParser()
    parser.add(
        '--config',
        is_config_file=True,
        default=os.path.join(p3d_path, 'config_sgp_debug.yml'),
        help='YAML config file path. Please refer to caps_config.yml as a '
        'reference. It overrides the default config file, but will be '
        'overridden by other command line inputs.')
    parser.add(
        '--no_verifier',
        type=bool,
        default=False,
        help='Overwrites overlap_thr to 0 for all iterations. No verifier will be used.')
    parser.add(
        '--n_epochs_per_new_label',
        type=int,
        default=None,
        help='If set, maps the current setting to a setting, where we generate every n epochs new pseudo labels. " \
        + "Overwrites meta_iters with sum(max_epoch_per_iter)//n, max_epoch_per_iter with [n]*meta_iters and expands overlap_thr to meta_iters.')
    config = parser.get_config()

    sgp = SGP3DRegistration()
    sgp.run(config)
