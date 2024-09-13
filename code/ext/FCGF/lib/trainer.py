# -*- coding: future_fstrings -*-
#
# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from FCGF (0612340)
#   (https://github.com/chrischoy/FCGF/tree/0612340ead256adb5449da8088f506e947e44b4c)
# Copyright (c) 2019 Chris Choy (chrischoy@ai.stanford.edu), Jaesik Park (jaesik.park@postech.ac.kr),
# licensed under the MIT license, cf. 3rd-party-licenses.txt file in the root directory of this
# source tree.

import os
import os.path as osp
import gc
import logging
import numpy as np
import json
import shutil

import sys
file_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(file_dir, "../../../geometry"))
sys.path.append(os.path.join(file_dir, "../../../perception3d"))
from adaptor import register
import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from model import load_model
import util.transform_estimation as te
from lib.metrics import pdist, corr_dist
from lib.timer import Timer, AverageMeter
from lib.eval import find_nn_gpu

from util.file import ensure_dir
from util.misc import _hash, decompose_tensors, cosine_scheduler
from util.pointcloud import get_matching_indices

import open3d as o3d
import MinkowskiEngine as ME
from geometry.pointcloud import solve, refine, make_o3d_pointcloud
from .knn import find_knn_batch
from .data_loaders import PairDataset

from collections import OrderedDict



class AlignmentTrainer:

  def __init__(
      self,
      config,
      data_loader,
      val_data_loader=None,
  ):
    num_feats = 1  # occupancy only for 3D Match dataset. For ScanNet, use RGB 3 channels.

    # Model initialization
    Model = load_model(config.model)
    model = Model(
        num_feats,
        config.model_n_out,
        bn_momentum=config.bn_momentum,
        normalize_feature=config.normalize_feature,
        conv1_kernel_size=config.conv1_kernel_size,
        D=3)

    if config.weights:
      checkpoint = torch.load(config.weights)
      model.load_state_dict(checkpoint['state_dict'])

    logging.info(model)

    self.config = config
    self.model = model
    self.max_epoch = config.max_epoch
    self.save_freq = config.save_freq_epoch
    self.val_max_iter = config.val_max_iter
    self.val_epoch_freq = config.val_epoch_freq

    self.best_val_metric = config.best_val_metric
    self.best_val_epoch = -np.inf
    self.best_val = -np.inf

    if config.use_gpu and not torch.cuda.is_available():
      logging.warning('Warning: There\'s no CUDA support on this machine, '
                      'training is performed on CPU.')
      raise ValueError('GPU not available, but cuda flag set')

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.optimizer = getattr(optim, config.optimizer)(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay)

    self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.exp_gamma)

    self.start_epoch = 1
    self.checkpoint_dir = config.out_dir

    ensure_dir(self.checkpoint_dir)
    json.dump(
        config,
        open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
        indent=4,
        sort_keys=False)

    self.iter_size = config.iter_size
    self.batch_size = data_loader.batch_size
    self.data_loader = data_loader
    self.val_data_loader = val_data_loader

    self.test_valid = True if self.val_data_loader is not None else False
    self.log_step = int(np.sqrt(self.config.batch_size))
    self.model = self.model.to(self.device)
    self.writer = SummaryWriter(logdir=config.out_dir)

    if config.resume is not None:
      if osp.isfile(config.resume):
        logging.info("=> loading checkpoint '{}'".format(config.resume))
        state = torch.load(config.resume)
        self.start_epoch = state['epoch'] + 1
        model.load_state_dict(state['state_dict'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.optimizer.load_state_dict(state['optimizer'])

        best_state_path = config.resume.replace("checkpoint.pt", "best_val_checkpoint.pt")
        if osp.isfile(best_state_path):
          best_state = torch.load(best_state_path) # this step can be memory optimzed by loading it
                                                   # first and then loading the model
        else:
          best_state = state
        if 'best_val' in best_state.keys():
          self.best_val = best_state['best_val']
          self.best_val_epoch = best_state['best_val_epoch']
          self.best_val_metric = best_state['best_val_metric']
      else:
        raise ValueError(f"=> no checkpoint found at '{config.resume}'")

    if config.self_training and config.ema_teacher:
      self.teacher = Model(
        num_feats,
        config.model_n_out,
        bn_momentum=config.bn_momentum,
        normalize_feature=config.normalize_feature,
        conv1_kernel_size=config.conv1_kernel_size,
        D=3)
      self.teacher.to(self.device)
      self.teacher_update(0) # initialize teacher model with student model
      self.momentum_teacher = cosine_scheduler(config.momentum_teacher,
                                               1 if config.momentum_teacher > 0 else 0,
                                               self.max_epoch,
                                               len(self.data_loader) // self.iter_size)
    else:
      self.teacher = self.model

  def train(self):
    """
    Full training logic
    """
    # Baseline random feature performance
    if self.test_valid and self.config.resume is None:
      with torch.no_grad():
        val_dict = self._valid_epoch()

      for k, v in val_dict.items():
        self.writer.add_scalar(f'val/{k}', v, self.start_epoch-1)

    for epoch in range(self.start_epoch, self.max_epoch + 1):
      lr = self.scheduler.get_lr()
      logging.info(f" Epoch: {epoch}, LR: {lr}")
      self._train_epoch(epoch)
      self._save_checkpoint(epoch)
      self.scheduler.step()

      if self.test_valid and epoch % self.val_epoch_freq == 0:
        with torch.no_grad():
          val_dict = self._valid_epoch()

        for k, v in val_dict.items():
          self.writer.add_scalar(f'val/{k}', v, epoch)
        if self.best_val < val_dict[self.best_val_metric]:
          logging.info(
              f'Saving the best val model with {self.best_val_metric}: {val_dict[self.best_val_metric]}'
          )
          self.best_val = val_dict[self.best_val_metric]
          self.best_val_epoch = epoch
          self._save_checkpoint(epoch, 'best_val_checkpoint')
        else:
          logging.info(
              f'Current best val model with {self.best_val_metric}: {self.best_val} at epoch {self.best_val_epoch}'
          )
      if epoch % self.config.save_add_checkpoint_interval == 0:
        self._save_checkpoint(epoch, f'checkpoint_after_{epoch}epochs')
      if epoch % 50 == 0:
        model_path = os.path.join(self.checkpoint_dir, 'best_val_checkpoint.pth')
        if os.path.exists(model_path):
          shutil.copyfile(model_path,
            os.path.join(self.checkpoint_dir, f'best_val_checkpoint_after_{epoch}epochs.pth')
          )
        else:
          self._save_checkpoint(epoch, f'checkpoint_after_{epoch}epochs.pth')


  def _save_checkpoint(self, epoch, filename='checkpoint'):
    state = {
        'epoch': epoch,
        'state_dict': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'scheduler': self.scheduler.state_dict(),
        'config': self.config,
        'best_val': self.best_val,
        'best_val_epoch': self.best_val_epoch,
        'best_val_metric': self.best_val_metric
    }
    filename = os.path.join(self.checkpoint_dir, f'{filename}.pth')
    logging.info("Saving checkpoint: {} ...".format(filename))
    torch.save(state, filename)

  def find_pairs(self, F0, F1, len_batch):
    """
    The following function is from DGR (1f4871d)
      (https://github.com/chrischoy/DeepGlobalRegistration/tree/1f4871d4c616fa2d2dc6d04a4506bd7d6e593fb4)
    Copyright (c) 2020 Chris Choy (chrischoy@ai.stanford.edu), Wei Dong (weidong@andrew.cmu.edu),
    licensed under the MIT license, cf. 3rd-party-licenses.txt file in the root directory of this source tree.

    """
    nn_batch = find_knn_batch(F0,
                              F1,
                              len_batch,
                              nn_max_n=self.config.nn_max_n,
                              knn=self.config.inlier_knn,
                              return_distance=False,
                              search_method=self.config.knn_search_method)

    pred_pairs = []
    for nns, lens in zip(nn_batch, len_batch):
      pred_pair_ind0, pred_pair_ind1 = torch.arange(
          len(nns)).long()[:, None], nns.long().cpu()
      nn_pairs = []
      for j in range(nns.shape[1]):
        nn_pairs.append(
            torch.cat((pred_pair_ind0.cpu(), pred_pair_ind1[:, j].unsqueeze(1)), 1))

      pred_pairs.append(torch.cat(nn_pairs, 0))
    return pred_pairs


class ContrastiveLossTrainer(AlignmentTrainer):

  def __init__(
      self,
      config,
      data_loader,
      val_data_loader=None,
  ):
    if val_data_loader is not None:
      assert val_data_loader.batch_size == 1, "Val set batch size must be 1 for now."
    AlignmentTrainer.__init__(self, config, data_loader, val_data_loader)
    self.neg_thresh = config.neg_thresh
    self.pos_thresh = config.pos_thresh
    self.neg_weight = config.neg_weight

  @staticmethod
  def apply_transform(pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    return pts @ R.t() + T

  def generate_rand_negative_pairs(self, positive_pairs, hash_seed, N0, N1, N_neg=0):
    """
    Generate random negative pairs
    """
    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)
    if N_neg < 1:
      N_neg = positive_pairs.shape[0] * 2
    pos_keys = _hash(positive_pairs, hash_seed)

    neg_pairs = np.floor(np.random.rand(int(N_neg), 2) * np.array([[N0, N1]])).astype(
        np.int64)
    neg_keys = _hash(neg_pairs, hash_seed)
    mask = np.isin(neg_keys, pos_keys, assume_unique=False)
    return neg_pairs[np.logical_not(mask)]

  def _train_epoch(self, epoch):
    gc.collect()
    self.model.train()
    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0

    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()

    iter_size = self.iter_size
    start_iter = (epoch - 1) * (len(data_loader) // iter_size)

    data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()

    # Main training
    for curr_iter in range(len(data_loader) // iter_size):
      self.optimizer.zero_grad()
      batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0

      data_time = 0
      total_timer.tic()
      for iter_idx in range(iter_size):
        # Caffe iter size
        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_time += data_timer.toc(average=False)

        # pairs consist of (xyz1 index, xyz0 index)
        sinput0 = ME.SparseTensor(
            input_dict['sinput0_F'].to(self.device),
            coordinates=input_dict['sinput0_C'].to(self.device))
        F0 = self.model(sinput0).F

        sinput1 = ME.SparseTensor(
            input_dict['sinput1_F'].to(self.device),
            coordinates=input_dict['sinput1_C'].to(self.device))
        F1 = self.model(sinput1).F

        N0, N1 = len(sinput0), len(sinput1)

        pos_pairs = input_dict['correspondences']
        neg_pairs = self.generate_rand_negative_pairs(pos_pairs, max(N0, N1), N0, N1)
        pos_pairs = pos_pairs.long().to(self.device)
        neg_pairs = torch.from_numpy(neg_pairs).long().to(self.device)

        neg0 = F0.index_select(0, neg_pairs[:, 0])
        neg1 = F1.index_select(0, neg_pairs[:, 1])
        pos0 = F0.index_select(0, pos_pairs[:, 0])
        pos1 = F1.index_select(0, pos_pairs[:, 1])

        # Positive loss
        pos_loss = (pos0 - pos1).pow(2).sum(1)

        # Negative loss
        neg_loss = F.relu(self.neg_thresh -
                          ((neg0 - neg1).pow(2).sum(1) + 1e-4).sqrt()).pow(2)

        pos_loss_mean = pos_loss.mean() / iter_size
        neg_loss_mean = neg_loss.mean() / iter_size

        # Weighted loss
        loss = pos_loss_mean + self.neg_weight * neg_loss_mean
        loss.backward(
        )  # To accumulate gradient, zero gradients only at the begining of iter_size
        batch_loss += loss.item()
        batch_pos_loss += pos_loss_mean.item()
        batch_neg_loss += neg_loss_mean.item()

      self.optimizer.step()

      torch.cuda.empty_cache()

      total_loss += batch_loss
      total_num += 1.0
      total_timer.toc()
      data_meter.update(data_time)

      # Print logs
      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/pos_loss', batch_pos_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/neg_loss', batch_neg_loss, start_iter + curr_iter)
        logging.info(
            "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f}"
            .format(epoch, curr_iter,
                    len(self.data_loader) //
                    iter_size, batch_loss, batch_pos_loss, batch_neg_loss) +
            "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(
                data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg))
        data_meter.reset()
        total_timer.reset()

  def _valid_epoch(self):
    # Change the network to evaluation mode
    self.model.eval()
    self.val_data_loader.dataset.reset_seed(0)
    num_data = 0
    hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter, reg_recall = AverageMeter(
    ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    data_timer, feat_timer, matching_timer = Timer(), Timer(), Timer()
    tot_num_data = len(self.val_data_loader.dataset)
    if self.val_max_iter > 0:
      tot_num_data = min(self.val_max_iter, tot_num_data)
    data_loader_iter = self.val_data_loader.__iter__()

    for batch_idx in range(tot_num_data):
      data_timer.tic()
      input_dict = data_loader_iter.next()
      data_timer.toc()

      # pairs consist of (xyz1 index, xyz0 index)
      feat_timer.tic()
      sinput0 = ME.SparseTensor(
          input_dict['sinput0_F'].to(self.device),
          coordinates=input_dict['sinput0_C'].to(self.device))
      F0 = self.model(sinput0).F

      sinput1 = ME.SparseTensor(
          input_dict['sinput1_F'].to(self.device),
          coordinates=input_dict['sinput1_C'].to(self.device))
      F1 = self.model(sinput1).F
      feat_timer.toc()

      matching_timer.tic()
      if self.config.self_training:
        with torch.no_grad():
          T_gt, _, _, _ = self.est_transf(input_dict)
      else:
        T_gt = input_dict['T_gt']

      xyz0, xyz1 = input_dict['pcd0'], input_dict['pcd1']
      xyz0_corr, xyz1_corr = self.find_corr(xyz0, xyz1, F0, F1, subsample_size=5000)
      T_est = te.est_quad_linear_robust(xyz0_corr, xyz1_corr)

      loss = corr_dist(T_est, T_gt, xyz0, xyz1, weight=None)
      loss_meter.update(loss)

      rte = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
      rte_meter.update(rte)
      rre = np.abs(np.arccos(np.clip((np.trace(T_est[:3, :3].t() @ T_gt[:3, :3]) - 1) / 2.0,
                                     -0.999999, 0.999999))) / np.pi * 180

      rre_meter.update(rre)
      reg_recall.update(rte < 0.3 and rre < 15)

      hit_ratio = self.evaluate_hit_ratio(
          xyz0_corr, xyz1_corr, T_gt, thresh=self.config.hit_ratio_thresh)
      hit_ratio_meter.update(hit_ratio)
      feat_match_ratio.update(hit_ratio > 0.05)
      matching_timer.toc()

      num_data += 1
      torch.cuda.empty_cache()

      if batch_idx % 100 == 0 and batch_idx > 0:
        logging.info(' '.join([
            f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f},",
            f"Feature Extraction Time: {feat_timer.avg:.3f}, Matching Time: {matching_timer.avg:.3f},",
            f"Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
            f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}",
            f"Reg Recall: {reg_recall.avg:.3f}"
        ]))
        data_timer.reset()

    logging.info(' '.join([
        f"Final Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
        f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}",
        f"Reg Recall: {reg_recall.avg:.3f}"
    ]))
    return {
        "loss": loss_meter.avg,
        "rre": rre_meter.avg,
        "rte": rte_meter.avg,
        'feat_match_ratio': feat_match_ratio.avg,
        'hit_ratio': hit_ratio_meter.avg,
        'reg_recall': reg_recall.avg
    }

  def find_corr(self, xyz0, xyz1, F0, F1, subsample_size=-1):
    subsample = len(F0) > subsample_size
    if subsample_size > 0 and subsample:
      N0 = min(len(F0), subsample_size)
      N1 = min(len(F1), subsample_size)
      inds0 = np.random.choice(len(F0), N0, replace=False)
      inds1 = np.random.choice(len(F1), N1, replace=False)
      F0, F1 = F0[inds0], F1[inds1]

    # Compute the nn
    nn_inds = find_nn_gpu(F0, F1, nn_max_n=self.config.nn_max_n)
    if subsample_size > 0 and subsample:
      return xyz0[inds0], xyz1[inds1[nn_inds]]
    else:
      return xyz0, xyz1[nn_inds]

  @staticmethod
  def evaluate_hit_ratio(xyz0, xyz1, T_gth, thresh=0.1):
    xyz0 = ContrastiveLossTrainer.apply_transform(xyz0, T_gth)
    dist = np.sqrt(((xyz0 - xyz1)**2).sum(1) + 1e-6)
    return (dist < thresh).float().mean().item()


class HardestContrastiveLossTrainer(ContrastiveLossTrainer):

  def contrastive_hardest_negative_loss(self,
                                        F0,
                                        F1,
                                        pos_pairs,
                                        neg0_pairs=None,
                                        neg1_pairs=None,
                                        num_pos_samples=5192,
                                        num_hn_samples=2048,
                                        thresh=None):
    """
    Generate hardest negative pairs
    """
    def sample_pair_features(F0, F1, pairs, max_samples, return_indices=False):
      if len(pairs) > max_samples:
        sel = np.random.choice(len(pairs), max_samples, replace=False)
        pairs = pairs[sel]
      ind0 = pairs[:, 0].long()
      ind1 = pairs[:, 1].long()
      if return_indices:
        return F0[ind0], F1[ind1], ind0, ind1
      else:
        return F0[ind0], F1[ind1]

    posF0, posF1, pos_ind0, pos_ind1 = sample_pair_features(F0, F1, pos_pairs,
                                                            max_samples=num_pos_samples,
                                                            return_indices=True)

    N0, N1 = len(F0), len(F1)
    hash_seed = max(N0, N1)
    sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)
    sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)
    # Find negatives for all F1[positive_pairs[:, 1]]
    subF0, subF1 = F0[sel0], F1[sel1]

    if not isinstance(pos_pairs, np.ndarray):
      pos_pairs = np.array(pos_pairs, dtype=np.int64)

    pos_keys = _hash(pos_pairs, hash_seed)

    if neg0_pairs is None:
      D01 = pdist(posF0, subF1, dist_type='L2')
      D01min, D01ind = D01.min(1)
      D01ind = sel1[D01ind.cpu().numpy()]
      neg_keys0 = _hash([pos_ind0.numpy(), D01ind], hash_seed)

      mask0 = torch.from_numpy(
          np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
      D01min = D01min[mask0]
    else:
      neg0F0, neg0F1 = sample_pair_features(F0, F1, neg0_pairs, max_samples=num_pos_samples)
      D01min = (neg0F0 - neg0F1).pow(2).sum(1)

    if neg1_pairs is None:
      D10 = pdist(posF1, subF0, dist_type='L2')
      D10min, D10ind = D10.min(1)
      D10ind = sel0[D10ind.cpu().numpy()]
      neg_keys1 = _hash([D10ind, pos_ind1.numpy()], hash_seed)

      mask1 = torch.from_numpy(
          np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
      D10min = D10min[mask1]
    else:
      neg1F0, neg1F1 = sample_pair_features(F0, F1, neg1_pairs, max_samples=num_pos_samples)
      D10min = (neg1F1 - neg1F0).pow(2).sum(1)
    

    pos_loss = F.relu((posF0 - posF1).pow(2).sum(1) - self.pos_thresh)
    neg_loss0 = F.relu(self.neg_thresh - D01min).pow(2)
    neg_loss1 = F.relu(self.neg_thresh - D10min).pow(2)
    return pos_loss.mean(), (neg_loss0.mean() + neg_loss1.mean()) / 2

  def est_transf(self, input_dict, feat="FCGF", fit_thresh:float=0):
    """Estimate transformation between two point clouds using self-training.
    Returns hardest negatives for point cloud 0 as a byproduct of NN(feat_space)."""

    # RANSAC+Refinement+NN(coord_space)
    xyz0_decomposed, xyz1_decomposed = decompose_tensors(input_dict["pcd0"], input_dict["pcd1"], input_dict["len_batch"])
    T_batch, ransac_fitness_batch, icp_fitness_batch, pos_pairs_batch = [], [], [], []
    start_idx = np.zeros(2, dtype=int)
    counter = 0
    for xyz0, xyz1, xyz0_raw, xyz1_raw, T0, T1, pc_len in zip(xyz0_decomposed, xyz1_decomposed, input_dict["xyz0_raw"], input_dict["xyz1_raw"], input_dict["T0"], input_dict["T1"], input_dict["len_batch"]):
      pcd0_no_trans = make_o3d_pointcloud(xyz0_raw)
      pcd1_no_trans = make_o3d_pointcloud(xyz1_raw)
      
      T, ransac_fitness, icp_fitness, _ = register(pcd0_no_trans, pcd1_no_trans, feat, 'RANSAC', self.teacher, self.config)
      T = torch.tensor(T1 @ T @ np.linalg.inv(T0)) # map back to randomly rotated coordinates

      pcd0 = make_o3d_pointcloud(xyz0)
      pcd1 = make_o3d_pointcloud(xyz1)
      pos_pairs = get_matching_indices(pcd0, pcd1, T, self.config.voxel_size*2) #see SGP for '*2', FCGF uses *1 for ground truth
      T_batch.append(T)
      ransac_fitness_batch.append(ransac_fitness)
      icp_fitness_batch.append(icp_fitness)

      if len(pos_pairs) > 0:
        pos_pairs += start_idx
        if icp_fitness >= fit_thresh:
          # filter out positives with totoal low ICP fitness like in SGP with verifier
          pos_pairs = torch.tensor(pos_pairs)
          pos_pairs_batch.append(pos_pairs)
      start_idx += pc_len
      counter += 1

    T_batch = torch.cat(T_batch).to(torch.float32)
    ransac_fitness_batch = torch.tensor(ransac_fitness_batch) if all([fit is not None for fit in ransac_fitness_batch]) else None
    icp_fitness_batch = torch.tensor(icp_fitness_batch) if all([fit is not None for fit in icp_fitness_batch]) else None
    pos_pairs_batch = torch.cat(pos_pairs_batch)

    return T_batch, ransac_fitness_batch, icp_fitness_batch, pos_pairs_batch

  @staticmethod
  def get_negatives(feat_pairs, pos_pairs):
    """Find negatives for positives from first point cloud
    """
    same_idx0 = feat_pairs[:, 0:1] == pos_pairs[:, 0]  # Comparing first elements of each pair
    diff_idx1 = feat_pairs[:, 1:2] != pos_pairs[:, 1]  # Comparing second elements of each pair
    neg0_idx0 = (same_idx0 & diff_idx1).nonzero()[:,0]
    neg01 = feat_pairs[neg0_idx0]

    return neg01


  def _train_epoch(self, epoch):
    gc.collect()
    self.model.train()
    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0
    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()
    iter_size = self.iter_size
    data_meter, data_timer, train_timer, total_timer = AverageMeter(), Timer(), Timer(), Timer()

    st_timer = {
      "nn_feat_space": Timer(),
      "ransac": Timer(),
      "refinement": Timer(),
      "nn_coord_space": Timer()
    }
    st_ransac_fitness, st_icp_fitness = AverageMeter(), AverageMeter()

    start_iter = (epoch - 1) * (len(data_loader) // iter_size)
    for curr_iter in range(len(data_loader) // iter_size):
      total_iter = start_iter + curr_iter
      self.optimizer.zero_grad()
      batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0

      data_time = 0
      total_timer.tic()
      for iter_idx in range(iter_size):
        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_time += data_timer.toc(average=False)

        sinput0 = ME.SparseTensor(
            input_dict['sinput0_F'].to(self.device),
            coordinates=input_dict['sinput0_C'].to(self.device))
        F0 = self.model(sinput0).F

        sinput1 = ME.SparseTensor(
            input_dict['sinput1_F'].to(self.device),
            coordinates=input_dict['sinput1_C'].to(self.device))

        F1 = self.model(sinput1).F

        if self.config.self_training:
          with torch.no_grad():
            feat = "FCGF"
            if self.config.sgp_bootstrap and epoch <= self.max_epoch * 0.1:
              feat = "FPFH"
            _, ransac_fitness, icp_fitness, pos_pairs = self.est_transf(
              input_dict, feat, fit_thresh=self.config.sgp_fit_thresh)
            if icp_fitness is not None:
              st_icp_fitness.update(icp_fitness.mean().item())
            if ransac_fitness is not None:
              st_ransac_fitness.update(ransac_fitness.mean().item())
        else:
          pos_pairs = input_dict['correspondences']
        
        train_timer.tic()
        pos_loss, neg_loss = self.contrastive_hardest_negative_loss(
            F0,
            F1,
            pos_pairs,
            neg0_pairs=None,
            neg1_pairs=None,
            num_pos_samples=self.config.num_pos_per_batch * self.config.batch_size,
            num_hn_samples=self.config.num_hn_samples_per_batch *
            self.config.batch_size)

        pos_loss /= iter_size
        neg_loss /= iter_size
        loss = pos_loss + self.neg_weight * neg_loss
        loss.backward()

        batch_loss += loss.item()
        batch_pos_loss += pos_loss.item()
        batch_neg_loss += neg_loss.item()
        train_timer.toc()

      self.optimizer.step()

      if self.config.self_training and self.config.ema_teacher:
        self.teacher_update(self.momentum_teacher[total_iter])

      gc.collect()

      torch.cuda.empty_cache()

      total_loss += batch_loss
      total_num += 1.0
      total_timer.toc()
      data_meter.update(data_time)

      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss, total_iter)
        self.writer.add_scalar('train/pos_loss', batch_pos_loss, total_iter)
        self.writer.add_scalar('train/neg_loss', batch_neg_loss, total_iter)
        if self.config.self_training:
          if st_ransac_fitness.count > 0:
            self.writer.add_scalar('train/ransac_fitness', st_ransac_fitness.avg, total_iter)
          if st_icp_fitness.count > 0:
            self.writer.add_scalar('train/icp_fitness', st_icp_fitness.avg, total_iter)
          logging.info(
              "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f}"
              .format(epoch, curr_iter,
                      len(self.data_loader) //
                      iter_size, batch_loss, batch_pos_loss, batch_neg_loss) +
              "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(
                  data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg))
          #logging.info(f"NN(feat) time: {st_timer['nn_feat_space'].avg:.1f}, RANSAC: {st_timer['ransac'].avg*self.config.batch_size:.2f}, Refine: {st_timer['refinement'].avg*self.config.batch_size:.2f}, NN(coord): {st_timer['nn_coord_space'].avg*self.config.batch_size:.2f}")
        else:
          logging.info(
              "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f}"
              .format(epoch, curr_iter,
                      len(self.data_loader) //
                      iter_size, batch_loss, batch_pos_loss, batch_neg_loss) +
              "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(
                  data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg))
        data_meter.reset()
        st_ransac_fitness.reset()
        st_icp_fitness.reset()
        total_timer.reset()

  def teacher_update(self, keep_rate):
    """
    Update the teacher model with the student model by using Exponential Moving Average (EMA)

    The method is from Unbiased Teacher (226f8c4)
      (https://github.com/facebookresearch/unbiased-teacher/blob/226f8c4b7b9c714e52f9bf640da7c75343803c52/ubteacher/engine/trainer.py#L632)
    Copyright (c) Facebook, Inc. and its affiliates., licensed under the MIT license,
    cf. 3rd-party-licenses.txt file in the root directory of this source tree.
    """
    student_model_dict = self.model.state_dict()

    new_teacher_dict = OrderedDict()
    for key, value in self.teacher.state_dict().items():
      if key in student_model_dict.keys():
        new_teacher_dict[key] = (
          student_model_dict[key] *
          (1 - keep_rate) + value * keep_rate
        )
      else:
        raise Exception("{} is not found in student model".format(key))

    self.teacher.load_state_dict(new_teacher_dict)
    for p in self.teacher.parameters():
      p.requires_grad = False
    self.teacher.to(self.device)
    self.teacher.eval()
    


class NonContrastiveLossTrainer(HardestContrastiveLossTrainer):

  def contrastive_hardest_negative_loss(self,
                                        F0,
                                        F1,
                                        pos_pairs,
                                        neg0_pairs=None,
                                        neg1_pairs=None,
                                        num_pos_samples=5192,
                                        num_hn_samples=2048,
                                        thresh=None):
    """
    Same as HardestConstrastiveLoss without negative pair loss
    """
    def sample_pair_features(F0, F1, pairs, max_samples, return_indices=False):
      if len(pairs) > max_samples:
        sel = np.random.choice(len(pairs), max_samples, replace=False)
        pairs = pairs[sel]
      ind0 = pairs[:, 0].long()
      ind1 = pairs[:, 1].long()
      if return_indices:
        return F0[ind0], F1[ind1], ind0, ind1
      else:
        return F0[ind0], F1[ind1]

    posF0, posF1 = sample_pair_features(F0, F1, pos_pairs, max_samples=num_pos_samples)
    
    pos_loss = F.relu((posF0 - posF1).pow(2).sum(1) - self.pos_thresh)
    return pos_loss.mean(), torch.tensor(0, dtype=float)
    


class TripletLossTrainer(ContrastiveLossTrainer):

  def triplet_loss(self,
                   F0,
                   F1,
                   positive_pairs,
                   num_pos=1024,
                   num_hn_samples=None,
                   num_rand_triplet=1024):
    """
    Generate negative pairs
    """
    N0, N1 = len(F0), len(F1)
    num_pos_pairs = len(positive_pairs)
    hash_seed = max(N0, N1)

    if num_pos_pairs > num_pos:
      pos_sel = np.random.choice(num_pos_pairs, num_pos, replace=False)
      sample_pos_pairs = positive_pairs[pos_sel]
    else:
      sample_pos_pairs = positive_pairs

    pos_ind0 = sample_pos_pairs[:, 0].long()
    pos_ind1 = sample_pos_pairs[:, 1].long()
    posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)

    pos_keys = _hash(positive_pairs, hash_seed)
    pos_dist = torch.sqrt((posF0 - posF1).pow(2).sum(1) + 1e-7)

    # Random triplets
    rand_inds = np.random.choice(
        num_pos_pairs, min(num_pos_pairs, num_rand_triplet), replace=False)
    rand_pairs = positive_pairs[rand_inds]
    negatives = np.random.choice(N1, min(N1, num_rand_triplet), replace=False)

    # Remove positives from negatives
    rand_neg_keys = _hash([rand_pairs[:, 0], negatives], hash_seed)
    rand_mask = np.logical_not(np.isin(rand_neg_keys, pos_keys, assume_unique=False))
    anchors, positives = rand_pairs[torch.from_numpy(rand_mask)].T
    negatives = negatives[rand_mask]

    rand_pos_dist = torch.sqrt((F0[anchors] - F1[positives]).pow(2).sum(1) + 1e-7)
    rand_neg_dist = torch.sqrt((F0[anchors] - F1[negatives]).pow(2).sum(1) + 1e-7)

    loss = F.relu(rand_pos_dist + self.neg_thresh - rand_neg_dist).mean()

    return loss, pos_dist.mean(), rand_neg_dist.mean()

  def _train_epoch(self, epoch):
    config = self.config

    gc.collect()
    self.model.train()

    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0
    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()
    iter_size = self.iter_size
    data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
    pos_dist_meter, neg_dist_meter = AverageMeter(), AverageMeter()
    start_iter = (epoch - 1) * (len(data_loader) // iter_size)
    for curr_iter in range(len(data_loader) // iter_size):
      self.optimizer.zero_grad()
      batch_loss = 0
      data_time = 0
      total_timer.tic()
      for iter_idx in range(iter_size):
        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_time += data_timer.toc(average=False)

        # pairs consist of (xyz1 index, xyz0 index)
        sinput0 = ME.SparseTensor(
            input_dict['sinput0_F'].to(self.device),
            coordinates=input_dict['sinput0_C'].to(self.device))
        F0 = self.model(sinput0).F

        sinput1 = ME.SparseTensor(
            input_dict['sinput1_F'].to(self.device),
            coordinates=input_dict['sinput1_C'].to(self.device))
        F1 = self.model(sinput1).F

        pos_pairs = input_dict['correspondences']
        loss, pos_dist, neg_dist = self.triplet_loss(
            F0,
            F1,
            pos_pairs,
            num_pos=config.triplet_num_pos * config.batch_size,
            num_hn_samples=config.triplet_num_hn * config.batch_size,
            num_rand_triplet=config.triplet_num_rand * config.batch_size)
        loss /= iter_size
        loss.backward()
        batch_loss += loss.item()
        pos_dist_meter.update(pos_dist)
        neg_dist_meter.update(neg_dist)

      self.optimizer.step()
      gc.collect()

      torch.cuda.empty_cache()

      total_loss += batch_loss
      total_num += 1.0
      total_timer.toc()
      data_meter.update(data_time)

      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
        logging.info(
            "Train Epoch: {} [{}/{}], Current Loss: {:.3e}, Pos dist: {:.3e}, Neg dist: {:.3e}"
            .format(epoch, curr_iter,
                    len(self.data_loader) //
                    iter_size, batch_loss, pos_dist_meter.avg, neg_dist_meter.avg) +
            "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(
                data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg))
        pos_dist_meter.reset()
        neg_dist_meter.reset()
        data_meter.reset()
        total_timer.reset()


class HardestTripletLossTrainer(TripletLossTrainer):

  def triplet_loss(self,
                   F0,
                   F1,
                   positive_pairs,
                   num_pos=1024,
                   num_hn_samples=512,
                   num_rand_triplet=1024):
    """
    Generate negative pairs
    """
    N0, N1 = len(F0), len(F1)
    num_pos_pairs = len(positive_pairs)
    hash_seed = max(N0, N1)
    sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)
    sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)

    if num_pos_pairs > num_pos:
      pos_sel = np.random.choice(num_pos_pairs, num_pos, replace=False)
      sample_pos_pairs = positive_pairs[pos_sel]
    else:
      sample_pos_pairs = positive_pairs

    # Find negatives for all F1[positive_pairs[:, 1]]
    subF0, subF1 = F0[sel0], F1[sel1]

    pos_ind0 = sample_pos_pairs[:, 0].long()
    pos_ind1 = sample_pos_pairs[:, 1].long()
    posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

    D01 = pdist(posF0, subF1, dist_type='L2')
    D10 = pdist(posF1, subF0, dist_type='L2')

    D01min, D01ind = D01.min(1)
    D10min, D10ind = D10.min(1)

    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)

    pos_keys = _hash(positive_pairs, hash_seed)

    D01ind = sel1[D01ind.cpu().numpy()]
    D10ind = sel0[D10ind.cpu().numpy()]
    neg_keys0 = _hash([pos_ind0.numpy(), D01ind], hash_seed)
    neg_keys1 = _hash([D10ind, pos_ind1.numpy()], hash_seed)

    mask0 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
    mask1 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
    pos_dist = torch.sqrt((posF0 - posF1).pow(2).sum(1) + 1e-7)

    # Random triplets
    rand_inds = np.random.choice(
        num_pos_pairs, min(num_pos_pairs, num_rand_triplet), replace=False)
    rand_pairs = positive_pairs[rand_inds]
    negatives = np.random.choice(N1, min(N1, num_rand_triplet), replace=False)

    # Remove positives from negatives
    rand_neg_keys = _hash([rand_pairs[:, 0], negatives], hash_seed)
    rand_mask = np.logical_not(np.isin(rand_neg_keys, pos_keys, assume_unique=False))
    anchors, positives = rand_pairs[torch.from_numpy(rand_mask)].T
    negatives = negatives[rand_mask]

    rand_pos_dist = torch.sqrt((F0[anchors] - F1[positives]).pow(2).sum(1) + 1e-7)
    rand_neg_dist = torch.sqrt((F0[anchors] - F1[negatives]).pow(2).sum(1) + 1e-7)

    loss = F.relu(
        torch.cat([
            rand_pos_dist + self.neg_thresh - rand_neg_dist,
            pos_dist[mask0] + self.neg_thresh - D01min[mask0],
            pos_dist[mask1] + self.neg_thresh - D10min[mask1]
        ])).mean()

    return loss, pos_dist.mean(), (D01min.mean() + D10min.mean()).item() / 2
