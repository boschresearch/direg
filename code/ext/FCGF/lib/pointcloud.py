# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from SGP (63d33cc)
#   (https://github.com/theNded/SGP/tree/63d33cc8bffde53676d9c4800f4b11804b53b360)
# Copyright (c) 2021 Wei Dong and Heng Yang, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import open3d as o3d
import numpy as np
import torch

import MinkowskiEngine as ME
from scipy.spatial import cKDTree


def make_o3d_pointcloud(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def extract_feats(pcd, feature_type, voxel_size, model=None):
    xyz = np.asarray(pcd.points)
    _, sel = ME.utils.sparse_quantize(xyz,
                                      return_index=True,
                                      quantization_size=voxel_size)
    xyz = xyz[sel]
    pcd = make_o3d_pointcloud(xyz)

    if feature_type == 'FPFH':
        radius_normal = voxel_size * 2
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal,
                                                 max_nn=30))
        radius_feat = voxel_size * 5
        feat = o3d.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feat,
                                                 max_nn=100))
        # (N, 33)
        return pcd, feat.data.T

    elif feature_type == 'FCGF':
        DEVICE = torch.device('cuda')
        coords = ME.utils.batched_coordinates(
            [torch.floor(torch.from_numpy(xyz) / voxel_size).int()]).to(DEVICE)

        feats = torch.ones(coords.size(0), 1).to(DEVICE)
        sinput = ME.SparseTensor(feats, coordinates=coords)  # .to(DEVICE)

        # (N, 32)
        return pcd, model(sinput).F.detach().cpu().numpy()

    else:
        raise NotImplementedError(
            'Unimplemented feature type {}'.format(feature_type))


def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
    feat1tree = cKDTree(feat1)
    dists, nn_inds = feat1tree.query(feat0, k=knn, workers=-1)
    if return_distance:
        return nn_inds, dists
    else:
        return nn_inds


def match_feats(feat_src, feat_dst, mutual_filter=True, k=1):
    if not mutual_filter:
        nns01 = find_knn_cpu(feat_src, feat_dst, knn=1, return_distance=False)
        corres01_idx0 = np.arange(len(nns01)).squeeze()
        corres01_idx1 = nns01.squeeze()
        return np.stack((corres01_idx0, corres01_idx1)).T
    else:
        # for each feat in src, find its k=1 nearest neighbours
        nns01 = find_knn_cpu(feat_src, feat_dst, knn=1, return_distance=False)
        # for each feat in dst, find its k nearest neighbours
        nns10 = find_knn_cpu(feat_dst, feat_src, knn=k, return_distance=False)
        # find corrs
        num_feats = len(nns01)
        corres01 = []
        if k == 1:
            for i in range(num_feats):
                if i == nns10[nns01[i]]:
                    corres01.append([i, nns01[i]])
        else:
            for i in range(num_feats):
                if i in nns10[nns01[i]]:
                    corres01.append([i, nns01[i]])
        # print(
        #     f'Before mutual filter: {num_feats}, after mutual_filter with k={k}: {len(corres01)}.'
        # )

        # Fallback if mutual filter is too aggressive
        if len(corres01) < 10:
            nns01 = find_knn_cpu(feat_src,
                                 feat_dst,
                                 knn=1,
                                 return_distance=False)
            corres01_idx0 = np.arange(len(nns01)).squeeze()
            corres01_idx1 = nns01.squeeze()
            return np.stack((corres01_idx0, corres01_idx1)).T

        return np.asarray(corres01)


def weighted_procrustes(A, B, weights=None):
    num_pts = A.shape[1]
    if weights is None:
        weights = np.ones(num_pts)

    # compute weighted center
    A_center = A @ weights / np.sum(weights)
    B_center = B @ weights / np.sum(weights)

    # compute relative positions
    A_ref = A - A_center[:, np.newaxis]
    B_ref = B - B_center[:, np.newaxis]

    # compute rotation
    M = B_ref @ np.diag(weights) @ A_ref.T
    U, _, Vh = np.linalg.svd(M)
    S = np.identity(3)
    S[-1, -1] = np.linalg.det(U) * np.linalg.det(Vh)
    R = U @ S @ Vh

    # compute translation
    t = B_center - R @ A_center

    return R, t


def solve(src, dst, corres, solver_type, distance_thr, ransac_iters):
    if solver_type.startswith('RANSAC'):
        if ransac_iters == 0:
            return np.eye(4), 1.0
        corres = o3d.utility.Vector2iVector(corres)

        # use parameters from BUFFER paper
        similar_th = 0.8
        confidence = 0.999
        result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            src, dst, corres, distance_thr,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(similar_th),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_thr)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(ransac_iters, confidence))
        fitness = result.fitness * np.minimum(
            1.0,
            float(len(dst.points)) / float(len(src.points)))

        return result.transformation, fitness

    else:
        raise NotImplementedError(
            'Unimplemented solver type {}'.format(solver_type))


def refine(src, dst, ransac_T, distance_thr):
    result = o3d.pipelines.registration.registration_icp(
        src, dst, distance_thr, ransac_T,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    icp_T = result.transformation
    icp_fitness = result.fitness

    fitness = icp_fitness * np.minimum(
        1.0,
        float(len(dst.points)) / float(len(src.points)))

    return icp_T, fitness


def w_procrustes_dgr(X, Y, w=None, eps=np.finfo(np.float32).eps):
  """
  DGR weighted Procrustes Implementation
  
  X: torch tensor N x 3
  Y: torch tensor N x 3
  w: torch tensor N

  The following function is from DGR (1f4871d)
      (https://github.com/chrischoy/DeepGlobalRegistration/tree/1f4871d4c616fa2d2dc6d04a4506bd7d6e593fb4)
  Copyright (c) 2020 Chris Choy (chrischoy@ai.stanford.edu), Wei Dong (weidong@andrew.cmu.edu),
  licensed under the MIT license, cf. 3rd-party-licenses.txt file in the root directory of this source tree.
  """
  # https://ieeexplore.ieee.org/document/88573
  assert len(X) == len(Y)
  if w is not None:
    W1 = torch.abs(w).sum()
    w_norm = w / (W1 + eps)
    w_norm = w_norm[:, None]
  else:
    w_norm = torch.full((len(X), 1), 1 / (len(X) + eps)).to(X.device)

  mux = (w_norm * X).sum(0, keepdim=True)
  muy = (w_norm * Y).sum(0, keepdim=True)

  # Use CPU for small arrays
  Sxy = (Y - muy).t().mm(w_norm * (X - mux)).cpu().double()
  U, D, V = Sxy.svd()
  S = torch.eye(3).double()
  if U.det() * V.det() < 0:
    S[-1, -1] = -1

  R = U.mm(S.mm(V.t())).float()
  t = (muy.cpu().squeeze() - R.mm(mux.cpu().t()).squeeze()).float()
  
  T = torch.empty((4, 4), dtype=torch.float32)
  T[:3, :3] = R
  T[:3, 3] = t
  T[3, :] = torch.tensor([0, 0, 0, 1])
  return T



def sparsify_data(input_dict, F0=None, F1=None, percentage=1):
    """
    Sparsifies `input_dict` and optionally `F0` and `F1` by randomly
    selecting a `percentage` of points from each batch. Only
    'correspondences' entry is not implemented yet and therefore
    missing from the output.
    """

    # Get the batch sizes
    batch_sizes = input_dict['len_batch']

    # Calculate the number of elements to keep for each batch
    max_num_per_batch0 = (percentage * torch.tensor(input_dict['len_batch'])[:,0]).int()
    max_num_per_batch1 = (percentage * torch.tensor(input_dict['len_batch'])[:,1]).int()

    # Initialize the sparsified input_dict
    sinput_dict = {}

    # Initialize the start indices for each batch
    start_idx0 = 0
    start_idx1 = 0

    # Initialize the new batch sizes
    new_batch_sizes = []

    # Initialize sparsified F0 and F1
    sF0 = []
    sF1 = []

    for i in range(len(batch_sizes)):
        N0 = batch_sizes[i][0]
        N1 = batch_sizes[i][1]

        # Generate random subsets of indices for each batch
        indices0 = torch.randperm(N0)[:max_num_per_batch0[i]] + start_idx0
        indices1 = torch.randperm(N1)[:max_num_per_batch1[i]] + start_idx1

        # Sparsify the input_dict for each batch
        pc0_keys=['sinput0_F', 'sinput0_C', 'pcd0']
        pc1_keys=['sinput1_F', 'sinput1_C', 'pcd1']
        for key in pc0_keys:
            sinput_dict[key] = input_dict[key][indices0] if i == 0 else torch.cat((sinput_dict[key], input_dict[key][indices0]), 0)
        for key in pc1_keys:
            sinput_dict[key] = input_dict[key][indices1] if i == 0 else torch.cat((sinput_dict[key], input_dict[key][indices1]), 0)

        # Sparsify F0 and F1 for each batch
        if F0 is not None:
            sF0.append(F0[indices0])
        if F1 is not None:
            sF1.append(F1[indices1])

        # Update the new batch sizes
        new_batch_sizes.append([len(indices0), len(indices1)])

        # Update the start indices for the next batch
        start_idx0 += N0
        start_idx1 += N1

    # Update the 'len_batch' entry in sinput_dict
    sinput_dict['len_batch'] = new_batch_sizes
    sinput_dict["T_gt"] = input_dict["T_gt"] # Copy over

    # Concat sF0 and sF1 if they are not None
    sF0 = torch.cat(sF0) if F0 is not None else None
    sF1 = torch.cat(sF1) if F1 is not None else None

    return sinput_dict, sF0, sF1