from typing import Mapping, Optional, Any

from torch import nn, Tensor

def initialize_weights(layer: nn.Module):
  classname = layer.__class__.__name__

  if classname.find('Conv') != -1:
    nn.init.normal_(layer.weight.data, 0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    nn.init.normal_(layer.weight.data, 1.0, 0.02)
    nn.init.constant_(layer.bias.data, 0)

class Generator(nn.Module):
  def __init__(
      self,
      latent_vector_size: int,
      feature_map_size: int,
      channel_count: int,
      *,
      with_weights: Optional[Mapping[str, Any]] = None,
      with_init: bool = False
  ):
    super(Generator, self).__init__()
    self.main = nn.Sequential(
      nn.ConvTranspose2d(latent_vector_size, feature_map_size * 8, 4, 1, 0, bias=False),
      nn.BatchNorm2d(feature_map_size * 8),
      nn.ReLU(True),
      nn.ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(feature_map_size * 4),
      nn.ReLU(True),
      nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(feature_map_size * 2),
      nn.ReLU(True),
      nn.ConvTranspose2d(feature_map_size * 2, feature_map_size, 4, 2, 1, bias=False),
      nn.BatchNorm2d(feature_map_size),
      nn.ReLU(True),
      nn.ConvTranspose2d(feature_map_size, channel_count, 4, 2, 1, bias=False),
      nn.Tanh()
    )
    if with_init: self.apply(initialize_weights)
    if with_weights: self.load_state_dict(with_weights)

  def forward(self, x: Tensor) -> Tensor:
    return self.main(x)
