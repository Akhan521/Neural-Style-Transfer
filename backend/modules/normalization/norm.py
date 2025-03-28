
'''
This module contains our normalization function for the NST model.

VGG networks are trained on images with each channel normalized by the mean and standard deviation of the ImageNet dataset.
We'll use the same normalization statistics to preprocess our content and style images.

The mean and standard deviation values are as follows:
- Mean: [0.485, 0.456, 0.406]
- Standard deviation: [0.229, 0.224, 0.225]

We'll normalize our images using these values before passing them through the VGG19 model.
'''

import torch
import torch.nn as nn

# Our normalization statistics.
cnn_norm_mean = [0.485, 0.456, 0.406]
cnn_norm_std = [0.229, 0.224, 0.225]

class Normalization(nn.Module):

    def __init__(self, mean, std):

        # Initialize the module.
        super(Normalization, self).__init__()

        '''
        An image tensor has shape [B, C, H, W], where:
        - B: batch size
        - C: number of channels (3 for RGB images)
        - H: height of the image
        - W: width of the image

        Our mean and std. dev. tensors should have the shape [C, 1, 1] to work with the image tensor.
        '''
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, image):

        # Normalize the image tensor.
        return (image - self.mean) / self.std