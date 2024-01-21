import numpy as np
from torch import nn, Tensor

def initialize_weights(layer: nn.Module):
  if isinstance(layer, nn.Linear):
    nn.init.xavier_uniform_(layer.weight.data)
    if layer.bias is not None:
      nn.init.constant_(layer.bias.data, 0)

class Classifier(nn.Module):
  def __init__(self, img_shape, num_traits):
    super(Classifier, self).__init__()
    self.img_shape = img_shape
    self.num_traits = num_traits

    self.model = nn.Sequential(
      nn.Linear(int(np.prod(img_shape)), 512),
      nn.ReLU(),
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Linear(256, num_traits)
    )

  def forward(self, x):
    return self.model(x.view(x.size(0), -1))
