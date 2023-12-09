from torch.functional import F
from torch import nn

class Generator(nn.Module):
  def __init__(self, z_dim, image_dim):
    super(Generator, self).__init__()
    self.deconv1 = nn.ConvTranspose2d(z_dim, image_dim * 8, 4, 1, 0)
    self.deconv1_bn = nn.BatchNorm2d(image_dim * 8)
    self.deconv2 = nn.ConvTranspose2d(image_dim * 8, image_dim * 4, 4, 2, 1)
    self.deconv2_bn = nn.BatchNorm2d(image_dim * 4)
    self.deconv3 = nn.ConvTranspose2d(image_dim * 4, image_dim * 2, 4, 2, 1)
    self.deconv3_bn = nn.BatchNorm2d(image_dim * 2)
    self.deconv4 = nn.ConvTranspose2d(image_dim * 2, image_dim, 4, 2, 1)
    self.deconv4_bn = nn.BatchNorm2d(image_dim)
    self.deconv5 = nn.ConvTranspose2d(image_dim, 3, 4, 2, 1)

  def forward(self, input):
    x = F.leaky_relu(self.deconv1_bn(self.deconv1(input)), 0.2)
    x = F.leaky_relu(self.deconv2_bn(self.deconv2(x)), 0.2)
    x = F.leaky_relu(self.deconv3_bn(self.deconv3(x)), 0.2)
    x = F.leaky_relu(self.deconv4_bn(self.deconv4(x)), 0.2)
    x = F.tanh(self.deconv5(x))

    return x
