from torch import nn, Tensor

def initialize_weights(layer: nn.Module):
  if isinstance(layer, nn.Linear):
    nn.init.xavier_uniform_(layer.weight.data)
    if layer.bias is not None:
      nn.init.constant_(layer.bias.data, 0)

class Classifier(nn.Module):
  def __init__(self, latent_vector_size: int, num_attributes: int, *, with_init: bool = False):
    super(Classifier, self).__init__()
    self.num_attributes = num_attributes
    self.layers = nn.Sequential(
      nn.Linear(latent_vector_size, 512),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Dropout(0.1),
      nn.Linear(128, num_attributes),
      nn.Sigmoid()
    )
    if with_init: self.apply(initialize_weights)

  def forward(self, x: Tensor) -> Tensor:
    return self.layers(x)
