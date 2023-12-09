import torch
from torch import nn
from torchvision import transforms

hidden_dim = 64
image_channels = 3
batch_size = 128
max_epochs = 100

display_step = 5
n_epochs_stop = 5
image_size = 64

transform = transforms.Compose([
  transforms.Resize(image_size),
  transforms.CenterCrop(image_size),
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

class Classifier(nn.Module):
  def __init__(self, im_chan=3, hidden_dim=64, n_classes=40):
    super(Classifier, self).__init__()
    self.classif = nn.Sequential(
      self.make_classif_block(im_chan, hidden_dim),
      self.make_classif_block(hidden_dim, hidden_dim * 2),
      self.make_classif_block(hidden_dim * 2, hidden_dim * 4, stride=3),
      self.make_classif_block(hidden_dim * 4, n_classes, final_layer=True),
    )

  def make_classif_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
    '''
    Function to return a sequence of operations corresponding to a block of the classifier
    Parameters:
        input_channels: how many channels the input feature representation has
        output_channels: how many channels the output feature representation should have
        kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
        stride: the stride of the convolution
        final_layer: a boolean, true if it is the final layer and false otherwise
                  (affects activation and batchnorm)
    '''
    if not final_layer:
      return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size, stride),
        nn.BatchNorm2d(output_channels),
        nn.LeakyReLU(0.2, inplace=True),
      )
    else:
      return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size, stride),
        nn.Tanh()
      )

  def forward(self, image):
    '''
    Function for completing a forward pass of the classifier
    Parameters:
        image: a flattened image tensor with dimension (im_chan)
    '''
    classif_pred = self.classif(image)
    return classif_pred.view(len(classif_pred), -1)

lr = 0.0002
# Temp values for CelebA
# number of samples in the dataset
n_rows = 202599
# number of classes in the dataset
n_classes = 40
classifier_loss = nn.MSELoss()
classifier = Classifier(image_channels, hidden_dim, n_classes).to('cpu')
classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

def initialize_weights(layer: nn.Module):
  if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
    torch.nn.init.normal_(layer.weight, 0.0, 0.02)
  if isinstance(layer, nn.BatchNorm2d):
    torch.nn.init.normal_(layer.weight, 0.0, 0.02)
    torch.nn.init.constant_(layer.bias, 0)
classifier = classifier.apply(initialize_weights)

# train code
