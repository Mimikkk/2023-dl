import torch
from torch import nn

class LatentSpaceManipulator:
  def __init__(self, generator: nn.Module, classifier: nn.Module, device: torch.device):
    self.generator = generator
    self.classifier = classifier
    self.device = device

  def manipulate(self, latent_vector, target_traits, alpha=0.01, steps=100):
    self.generator.to(self.device)
    self.classifier.to(self.device)
    latent_vector = latent_vector.clone().detach().requires_grad_(True)
    latent_vector.to(self.device)
    target_traits.to(self.device)

    vectors = [latent_vector.clone()]
    for _ in range(steps):
      generated_image = self.generator(latent_vector)
      predicted_traits = self.classifier(generated_image)
      loss = ((predicted_traits - target_traits) ** 2).mean()
      loss.backward()

      with torch.no_grad():
        latent_vector -= alpha * latent_vector.grad
      latent_vector.grad.zero_()
      vectors.append(latent_vector.clone())

    return latent_vector, vectors