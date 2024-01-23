from typing import Callable, Optional

import torch
from torch import nn, Tensor

class LatentSpaceManipulator:
  def __init__(self, generator: nn.Module, classifier: nn.Module, device: torch.device, loss_fn: Callable[[Tensor, Tensor], Tensor]):
    self.generator = generator
    self.classifier = classifier
    self.device = device
    self.loss_fn = loss_fn

  def manipulate(
      self,
      latent_vector: Tensor,
      target_traits: Tensor,
      alpha: float = 0.01,
      steps: int = 100,
      on_step: Optional[Callable[[int, Tensor, Tensor, Tensor], None]] = None
  ):
    latent_vector = latent_vector.clone().detach().requires_grad_(True)
    self.classifier.to(self.device)
    self.generator.to(self.device)
    latent_vector.to(self.device)
    target_traits.to(self.device)

    for _ in range(steps):
      generated_image = self.generator(latent_vector)
      predicted_traits = self.classifier(generated_image)

      if  isinstance(self.loss_fn, nn.BCELoss):
        predicted_traits = predicted_traits.squeeze(0)
      loss = self.loss_fn(predicted_traits, target_traits)
      if  isinstance(self.loss_fn, nn.BCELoss):
        predicted_traits = predicted_traits.unsqueeze(0)

      loss.backward()

      with torch.no_grad(): latent_vector -= alpha * latent_vector.grad
      latent_vector.grad.zero_()

      if on_step: on_step(_, generated_image, predicted_traits, latent_vector.clone())

    return latent_vector
