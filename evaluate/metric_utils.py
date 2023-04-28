from copy import deepcopy
import torchmetrics
from pytorch_lightning.utilities.apply_func import move_data_to_device

import torch
from torch import nn
from torch.utils.data.dataloader import default_collate
import numpy as np

import heapq
from functools import partial


val_metrics = {
  'pos_diff_mean': torchmetrics.MeanMetric(),
  'x_diff_mean': torchmetrics.MeanMetric(),
  'y_diff_mean': torchmetrics.MeanMetric(),
  'yaw_diff_mean': torchmetrics.MeanMetric(),

  'collision_front': torchmetrics.MeanMetric(),
  'collision_rear': torchmetrics.MeanMetric(),
  'collision_side': torchmetrics.MeanMetric(),
  'collision_rate': torchmetrics.MeanMetric(),

  'off_ref': torchmetrics.MeanMetric(),
  'off_road': torchmetrics.MeanMetric(),

  'displacement_error_l2': torchmetrics.MeanMetric(),
  'distance_ref_trajectory': torchmetrics.MeanMetric(),

  'acc_mean': torchmetrics.MeanMetric(),
  'jerk_mean': torchmetrics.MeanMetric(),
  'comfort': torchmetrics.MeanMetric(),
}


class LazyEval:
  def __init__(self, func, *args, **kwargs):
    self.func = func
    self.args = args
    self.kwargs = kwargs

  def __repr__(self):
    return F"Func: {{{self.func}}}; args: {{{self.args}}}; kwargs: {{{self.kwargs}}}"

  @staticmethod
  def evaluate(lazy_obj):
    assert isinstance(lazy_obj, LazyEval), F"Can only evaluate LazyEval, but got {lazy_obj}"

    # nested lazy eval
    if isinstance(lazy_obj.func, LazyEval):
      lazy_obj.func = LazyEval.evaluate(lazy_obj.func)
    lazy_obj.args = [LazyEval.evaluate(arg) if isinstance(
        arg, LazyEval) else arg for arg in lazy_obj.args]
    lazy_obj.kwargs = {k: (LazyEval.evaluate(v) if isinstance(v, LazyEval) else v)
                       for k, v in lazy_obj.kwargs.items()}

    return lazy_obj.func(*lazy_obj.args, **lazy_obj.kwargs)

  def __call__(self):
    return self

  def __getattr__(self, attr):
    return partial(LazyEval, LazyEval(getattr, self, attr))


def get_device(val):
  if isinstance(val, torch.Tensor):
    return val.device
  elif isinstance(val, dict):
    for v in val.values():
      res = get_device(v)
      if res is not None:
        return res
  elif '__iter__' in dir(val) or isinstance(val, list) or isinstance(val, tuple):
    for v in val:
      res = get_device(v)
      if res is not None:
        return res

  return None


class DictMetric(torchmetrics.Metric):

  def __init__(self, metric_dict={}, prefix='', allow_auto_add=True, default_metrics=torchmetrics.MeanMetric):
    super().__init__()

    self.metric_dict = nn.ModuleDict(metric_dict)
    self._prefix = prefix
    self.allow_auto_add = allow_auto_add
    self.default_metrics = default_metrics

  @property
  def prefix(self):
    return self._prefix

  @prefix.setter
  def prefix(self, new_prefix):
    assert isinstance(new_prefix, str), F"Prefix needs to be str, instead got {new_prefix}"
    self._prefix = new_prefix

  def forward(self, update_metric_dict):
    if not self.allow_auto_add:
      # partial forward
      assert len(set(update_metric_dict.keys()) - set(self.metric_dict.keys())) == 0,\
          F"{set(update_metric_dict.keys())} not in {set(self.metric_dict.keys())}"

    res = {}
    for k, v in update_metric_dict.items():
      if self.allow_auto_add and k not in self.metric_dict:
        device = get_device(v)
        self.metric_dict[k] = self.default_metrics().to(device)
      res[self.prefix + k] = self.metric_dict[k](v)

    return res

  def update(self, update_metric_dict):
    if not self.allow_auto_add:
      # partial forward
      assert len(set(update_metric_dict.keys()) - set(self.metric_dict.keys())) == 0,\
          F"{set(update_metric_dict.keys())} not in {set(self.metric_dict.keys())}"

    for k, v in update_metric_dict.items():
      if self.allow_auto_add and k not in self.metric_dict:
        device = get_device(v)
        self.metric_dict[k] = self.default_metrics().to(device)
      self.metric_dict[k].update(v)

  def compute(self, keys=None):
    # partial compute
    if keys is None:
      res = {self.prefix + k: safe_compute(v) for k, v in self.metric_dict.items()}
    else:
      res = {self.prefix + k: safe_compute(self.metric_dict[k]) for k in keys}
    res = {k: v for k, v in res.items() if v is not None}
    return res

  def reset(self, keys=None):
    # partial compute
    if keys is None:
      return {self.prefix + k: v.reset() for k, v in self.metric_dict.items()}
    else:
      return {self.prefix + k: self.metric_dict[k].reset() for k in keys}

  def clone(self, prefix=''):
    metrics = deepcopy(self)
    metrics.prefix = prefix

    return metrics


class CategoricalMetric:
  pass


def make_categorical(cls: torchmetrics.Metric):
  class MyCategoricalMetric(torchmetrics.Metric, CategoricalMetric):

    def __init__(self, *args, **kwargs):
      super().__init__()

      self.metric_dict = nn.ModuleDict()
      self._make_new = partial(cls, *args, **kwargs)

    def update(self, cat, values=None):
      if values is None:
        cat, values = cat

      assert len(cat) == len(values), F"Lengths need to match. Got {len(cat)} and {len(values)}"

      for k in set(cat) - set(self.metric_dict.keys()):
        self.metric_dict[str(k)] = self._make_new()

      for c, v in zip(cat, values):
        self.metric_dict[str(c)].update(v)

    def reset(self, keys=None):
      # partial compute
      if keys is None:
        return {k: v.reset() for k, v in self.metric_dict.items()}
      else:
        return {k: self.metric_dict[k].reset() for k in keys}

    def compute(self):
      res = {k: safe_compute(v) for k, v in self.metric_dict.items()}
      res = {k: v for k, v in res.items() if v is not None}

      return res

    def clone(self, prefix=None):
      metrics = deepcopy(self)

      return metrics

  return MyCategoricalMetric


class SampleCatMetric(torchmetrics.CatMetric):

  def __init__(self, *args, prob=1.0, **kwargs):
    super().__init__(*args, **kwargs)

    self._prob = prob

  def update(self, value):
    # deal float
    if isinstance(value, float) and np.random.uniform() < self._prob:
      super().update(value)
    else:
      super().update(value[np.random.choice(len(value), int(
          np.ceil(self._prob * len(value))), replace=False)])


class HeapKRecord(torchmetrics.Metric):
  def __init__(self, top_k=20, mode='min'):
    super().__init__()

    self._top_k = top_k
    self._mode = mode

    self.add_state("loads", default=[], dist_reduce_fx=default_collate)
    self.add_state("scores", default=[], dist_reduce_fx=default_collate)

  def to(self, *args, **kwargs):
    super().to(*args, **kwargs)
    self.loads = move_data_to_device(self.loads, self.device)

  def update(self, scores, loads=None):
    if loads is None:
      scores, loads = scores

    # find top k
    comp = heapq.nsmallest if self._mode == 'min' else heapq.nlargest
    def heap_filter(items): return comp(self._top_k, items, key=lambda x: x[0])

    scores = self.scores + list(scores)
    loads = self.loads + list(loads)

    new_list = heap_filter(zip(scores, loads))
    new_list = list(zip(*new_list))

    self.scores = list(new_list[0])
    self.loads = list(new_list[1])

    # deal lazy evaluation
    # self.loads = [LazyEval.evaluate(load) if isinstance(load, LazyEval) else load for load in self.loads]
    for i, load in enumerate(self.loads):
      if isinstance(load, LazyEval):
        self.loads[i] = LazyEval.evaluate(load)

    self.loads = move_data_to_device(self.loads, self.device)

    # # unsqueeze tensor so perform stack instead of concat
    # if isinstance(self.loads[0], torch.Tensor):
    #     self.loads = [load.unsqueeze(0) for load in self.loads]

  def compute(self):

    if isinstance(self.loads, torch.Tensor):
      return self.loads
    else:
      return default_collate(self.loads)


def safe_compute(metric, default_val=None):
  try:
    return metric.compute()
  except Exception as e:
    return e