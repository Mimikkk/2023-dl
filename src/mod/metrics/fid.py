import numpy as np
import torch
from scipy.linalg import sqrtm


def calculate_fid(model, images_fake: torch.Tensor, images_real: torch.Tensor):
  features_fake = model(images_fake).detach().numpy()
  features_real = model(images_real).detach().numpy()

  average_fake = features_fake.mean(axis=0)
  covariance_fake = np.cov(features_fake, rowvar=False)
  average_real = features_real.mean(axis=0)
  covariance_real = np.cov(features_real, rowvar=False)

  averages_square_differences = np.sum((average_fake - average_real) ** 2.0)
  covariance_mean = sqrtm(covariance_fake.dot(covariance_real))

  if np.iscomplexobj(covariance_mean): covariance_mean = covariance_mean.real

  return (
      averages_square_differences
      + np.trace(covariance_fake + covariance_real - 2.0 * covariance_mean)
  )
