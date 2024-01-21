import torch
from torch import nn
from typing import Optional
import numpy as np

class EarlyStopping(object):
  def __init__(self, model: nn.Module, patience: int, verbose: bool = False, save_to: Optional[str] = None):
    self.patience = patience
    self.verbose = verbose
    self.save_to = save_to
    self.patience_count = 0
    self.best_loss = None
    self.last_loss = np.Inf
    self.should_stop = False
    self.model = model

  def step(self, loss: float):
    if self.best_loss is None:
      self.best_loss = loss
      if self.save_to: self._save_checkpoint(loss)
    elif loss > self.best_loss:
      self.patience_count += 1
      if self.verbose: print(f'EarlyStopping counter: {self.patience_count}/{self.patience}')
      if self.patience_count >= self.patience: self.should_stop = True
    else:
      self.best_loss = loss
      if self.save_to: self._save_checkpoint(loss)
      self.patience_count = 0
    if self.verbose and self.should_stop: print('EarlyStopping: stops due to lack of improvement')
    return self.should_stop

  def _save_checkpoint(self, loss: float):
    if self.verbose: print(f'Validation loss decreased ({self.last_loss:.6f} --> {loss:.6f}).  Saving model ...')
    torch.save(self.model.state_dict(), self.save_to)
    self.last_loss = loss
