# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from SGP (63d33cc)
#   (https://github.com/theNded/SGP/tree/63d33cc8bffde53676d9c4800f4b11804b53b360)
# Copyright (c) 2021 Wei Dong and Heng Yang, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.


import sys, os

file_path = os.path.abspath(__file__)
project_path = os.path.dirname(os.path.dirname(file_path))
sys.path.append(project_path)



from dataset.threedmatch_train import Dataset3DMatchTrain
from perception3d.adaptor import DatasetFCGFAdaptor, FCGFConfigParser, fcgf_train

if __name__ == '__main__':
    parser = FCGFConfigParser()
    parser.add('--max_epoch',
                type=int,
                default=25,
                help='Maximum number of epochs to train supervised or with self-training.')
    parser.add("--ema_teacher",
                type=bool,
                default=False,
                help="Use EMA teacher model for self-training.")
    parser.add('--momentum_teacher',
               type=float,
               default=0.99,
               help="""Base EMA parameter for teacher update. The value is increased to 1 during training with cosine schedule.""")
    parser.add('--use_negative_pair_byproducts',
                type=bool,
                default=False,
                help='In self-training, hard negatives (from pc0 to pc1) are generated as byproducts of positive pairs to optimize runtime.')
    parser.add("--sgp_bootstrap",
                type=bool,
                default=False,
                help="Use FPFH features for the first 10% of epochs of self-training.")
    parser.add("--sgp_fit_thresh",
                type=float,
                default=0,
                help="Threshold for ICP fitness to filter positive pairs like with a SGP verifier.")
    parser.add(
        '--config',
        is_config_file=True,
        default=os.path.join(os.path.dirname(__file__), 'config_train_debug.yml'),
        help='YAML config file path. Please refer to caps_config.yml as a '
        'reference. It overrides the default config file, but will be '
        'overridden by other command line inputs.')
    args_str = " ".join(sys.argv[1:])

    config = parser.get_config()
    dataset = DatasetFCGFAdaptor(
        Dataset3DMatchTrain(config.dataset_path, config.scenes, 0.3), config)
    if config.test_valid:
        val_dataset = DatasetFCGFAdaptor(
            Dataset3DMatchTrain(config.dataset_path, config.val_scenes, 0.3), config)
    else:
        val_dataset = None

    fcgf_train(dataset, val_dataset, config)
