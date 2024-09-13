# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from FCGF (0612340)
#   (https://github.com/chrischoy/FCGF/tree/0612340ead256adb5449da8088f506e947e44b4c)
# Copyright (c) 2019 Chris Choy (chrischoy@ai.stanford.edu), Jaesik Park (jaesik.park@postech.ac.kr),
# licensed under the MIT license, cf. 3rd-party-licenses.txt file in the root directory of this
# source tree.


import math
import time


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0.0
    self.sq_sum = 0.0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count
    self.sq_sum += val**2 * n
    self.var = self.sq_sum / self.count - self.avg**2


class Timer(object):
  """A simple timer."""

  def __init__(self, binary_fn=None, init_val=0):
    self.total_time = 0.
    self.calls = 0
    self.start_time = 0.
    self.diff = 0.
    self.binary_fn = binary_fn
    self.tmp = init_val

  def reset(self):
    self.total_time = 0
    self.calls = 0
    self.start_time = 0
    self.diff = 0

  @property
  def avg(self):
    return self.total_time / self.calls

  def tic(self):
    # using time.time instead of time.clock because time time.clock
    # does not normalize for multithreading
    self.start_time = time.time()

  def toc(self, average=True):
    return self.add(time.time() - self.start_time, average=average)

  def add(self, diff, average=True):
    self.diff = diff
    self.total_time += self.diff
    self.calls += 1
    if self.binary_fn:
      self.tmp = self.binary_fn(self.tmp, self.diff)
    if average:
      return self.avg
    else:
      return self.diff


class MinTimer(Timer):

  def __init__(self):
    Timer.__init__(self, binary_fn=lambda x, y: min(x, y), init_val=math.inf)

  @property
  def min(self):
    return self.tmp
