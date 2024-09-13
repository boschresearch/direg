# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from SGP (63d33cc)
#   (https://github.com/theNded/SGP/tree/63d33cc8bffde53676d9c4800f4b11804b53b360)
# Copyright (c) 2021 Wei Dong and Heng Yang, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.


from tqdm import tqdm

class SGPBase:
    def __init__(self):
        pass

    def teach_bootstrap(self, sgp_dataset, config):
        '''
        Teach without deep models. Use classical FPFH/SIFT features.
        '''
        for i, data in tqdm(enumerate(sgp_dataset)):
            src_data, dst_data, src_info, dst_info, pair_info = data

            label, pair_info = self.perception_bootstrap(
                src_data, dst_data, src_info, dst_info, config)
            sgp_dataset.write_pseudo_label(i, label, pair_info)

    def teach(self, sgp_dataset, model, config):
        '''
        Teach with deep models. Use learned FCGF/CAPS features.
        '''
        for i, data in tqdm(enumerate(sgp_dataset)):
            src_data, dst_data, src_info, dst_info, pair_info = data

            # if self.is_valid(src_info, dst_info, pair_info):
            label, pair_info = self.perception(src_data, dst_data, src_info,
                                               dst_info, model, config)
            sgp_dataset.write_pseudo_label(i, label, pair_info)

    def learn(self, sgp_dataset, config, val_dataset=None):
        # Adapt and dispatch training script to external implementations
        self.train_adaptor(sgp_dataset, config, val_dataset)

    # override
    def train_adaptor(self, sgp_dataset, config):
        pass

    # override
    def perception_bootstrap(self, src_data, dst_data, src_info, dst_info):
        pass

    # override
    def perception(self, src_data, dst_data, src_info, dst_info, model):
        pass

