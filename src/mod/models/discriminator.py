from typing import Optional, Mapping, Any

from torch import nn, Tensor

def initialize_weights(layer: nn.Module):
  classname = layer.__class__.__name__

  if classname.find('Conv') != -1:
    nn.init.normal_(layer.weight.data, 0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    nn.init.normal_(layer.weight.data, 1.0, 0.02)
    nn.init.constant_(layer.bias.data, 0)

class Discriminator(nn.Module):
  def __init__(
      self,
      channel_count: int,
      feature_map_size: int,
      *,
      with_init: bool = False,
      with_weights: Optional[Mapping[str, Any]] = None
  ):
    super(Discriminator, self).__init__()
    self.main = nn.Sequential(
      nn.Conv2d(channel_count, feature_map_size, 4, 2, 1, bias=False),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(feature_map_size, feature_map_size * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(feature_map_size * 2),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(feature_map_size * 2, feature_map_size * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(feature_map_size * 4),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(feature_map_size * 4, feature_map_size * 8, 4, 2, 1, bias=False),
      nn.BatchNorm2d(feature_map_size * 8),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(feature_map_size * 8, 1, 4, 1, 0, bias=False),
      nn.Sigmoid()
    )
    if with_init: self.apply(initialize_weights)
    if with_weights: self.load_state_dict(with_weights)

  def forward(self, input: Tensor) -> Tensor:
    return self.main(input)
