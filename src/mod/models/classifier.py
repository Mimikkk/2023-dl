from typing import Optional, Mapping, Any

from torch import nn

def initialize_weights(layer):
  if isinstance(layer, nn.Conv2d):
    nn.init.kaiming_normal_(layer.weight.data, mode='fan_out', nonlinearity='relu')
  elif isinstance(layer, nn.BatchNorm2d):
    nn.init.constant_(layer.weight, 1)
    nn.init.constant_(layer.bias, 0)
  elif isinstance(layer, nn.Linear):
    nn.init.normal_(layer.weight, 0, 0.01)
    nn.init.constant_(layer.bias, 0)

class Classifier(nn.Module):
  def __init__(
      self,
      image_shape: tuple[int, int, int],
      trait_count: int,
      *,
      with_weights: Optional[Mapping[str, Any]] = None,
      with_init: bool = False
  ):
    super(Classifier, self).__init__()
    (channel_count, width, height) = image_shape
    self.model = nn.Sequential(
      nn.Conv2d(channel_count, 32, 3, 1, 1),
      nn.ReLU(True),
      nn.MaxPool2d(2, 2),
      nn.Conv2d(32, 64, 3, 1, 1),
      nn.ReLU(True),
      nn.MaxPool2d(2, 2),
      nn.Conv2d(64, 128, 3, 1, 1),
      nn.ReLU(True),
      nn.MaxPool2d(2, 2),
      nn.Flatten(),
      nn.Linear(128 * (width // 8) * (height // 8), 512),
      nn.ReLU(True),
      nn.Dropout(0.5),
      nn.Linear(512, trait_count),
      nn.Sigmoid()
    )

    if with_init:
      self.apply(initialize_weights)
    if with_weights:
      self.load_state_dict(with_weights)

  def forward(self, x):
    return self.model(x)
