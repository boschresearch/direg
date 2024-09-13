# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from FCGF (0612340)
#   (https://github.com/chrischoy/FCGF/tree/0612340ead256adb5449da8088f506e947e44b4c)
# Copyright (c) 2019 Chris Choy (chrischoy@ai.stanford.edu), Jaesik Park (jaesik.park@postech.ac.kr),
# licensed under the MIT license, cf. 3rd-party-licenses.txt file in the root directory of this
# source tree.


import torch
import numpy as np
import MinkowskiEngine as ME


def _hash(arr, M):
  if isinstance(arr, np.ndarray):
    N, D = arr.shape
  else:
    N, D = len(arr[0]), len(arr)

  hash_vec = np.zeros(N, dtype=np.int64)
  for d in range(D):
    if isinstance(arr, np.ndarray):
      hash_vec += arr[:, d] * M**d
    else:
      hash_vec += arr[d] * M**d
  return hash_vec


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
  """
  The following function is from DINO, licensed under the Apache-2.0 license, cf. 3rd-party-licenses.txt
  file in the root directory of this source tree.
    (https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/utils.py#L187)
  """
  warmup_schedule = np.array([])
  warmup_iters = warmup_epochs * niter_per_ep
  if warmup_epochs > 0:
      warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

  iters = np.arange(epochs * niter_per_ep - warmup_iters)
  schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

  schedule = np.concatenate((warmup_schedule, schedule))
  assert len(schedule) == epochs * niter_per_ep
  return schedule


def decompose_tensors(tens1, tens2, len_batch):
    decomposed_tens1 = []
    decomposed_tens2 = []
    start_ind1 = 0
    start_ind2 = 0
    for len1, len2 in len_batch:
        decomposed_tens1.append(tens1[start_ind1 : start_ind1 + len1])
        start_ind1 += len1
        decomposed_tens2.append(tens2[start_ind2 : start_ind2 + len2])
        start_ind2 += len2
    return decomposed_tens1, decomposed_tens2


def extract_features(model,
                     xyz,
                     rgb=None,
                     normal=None,
                     voxel_size=0.05,
                     device=None,
                     skip_check=False,
                     is_eval=True):
  '''
  xyz is a N x 3 matrix
  rgb is a N x 3 matrix and all color must range from [0, 1] or None
  normal is a N x 3 matrix and all normal range from [-1, 1] or None

  if both rgb and normal are None, we use Nx1 one vector as an input

  if device is None, it tries to use gpu by default

  if skip_check is True, skip rigorous checks to speed up

  model = model.to(device)
  xyz, feats = extract_features(model, xyz)
  '''
  if is_eval:
    model.eval()

  if not skip_check:
    assert xyz.shape[1] == 3

    N = xyz.shape[0]
    if rgb is not None:
      assert N == len(rgb)
      assert rgb.shape[1] == 3
      if np.any(rgb > 1):
        raise ValueError('Invalid color. Color must range from [0, 1]')

    if normal is not None:
      assert N == len(normal)
      assert normal.shape[1] == 3
      if np.any(normal > 1):
        raise ValueError('Invalid normal. Normal must range from [-1, 1]')

  if device is None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  feats = []
  if rgb is not None:
    # [0, 1]
    feats.append(rgb - 0.5)

  if normal is not None:
    # [-1, 1]
    feats.append(normal / 2)

  if rgb is None and normal is None:
    feats.append(np.ones((len(xyz), 1)))

  feats = np.hstack(feats)

  # Voxelize xyz and feats
  coords = np.floor(xyz / voxel_size)
  coords, inds = ME.utils.sparse_quantize(coords, return_index=True)
  # Convert to batched coords compatible with ME
  coords = ME.utils.batched_coordinates([coords])
  return_coords = xyz[inds]

  feats = feats[inds]

  feats = torch.tensor(feats, dtype=torch.float32)
  coords = torch.tensor(coords, dtype=torch.int32)

  stensor = ME.SparseTensor(feats, coordinates=coords, device=device)

  return return_coords, model(stensor).F
