
'''
This file contains the content loss function used in the NST model.

Given a generated image and a content image, we observe their feature maps at a certain layer in the VGG19 model.
The content loss is the mean squared error between the feature maps of the generated image and the content image.
Our goal is to minimize this loss to ensure that the generated image captures the content of the content image.

PLAN: We'll add this module directly after the convolution layers that produce our feature maps.
This will allow us to calculate the content loss at the desired layers in the VGG19 model.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContentLoss(nn.Module):
    
    def __init__(self, target,):
        # Initialize the module.
        super(ContentLoss, self).__init__()

        # To prevent PyTorch from tracking the target tensor's gradients, we 'detach' it from the computation graph.
        # This ensures that the target tensor is treated as a constant.
        self.target = target.detach()

    def forward(self, input):
        # Compute the MSE loss between the target tensor and the input tensor.
        self.loss = F.mse_loss(input, self.target)
        
        # Since this module is only being used to compute content loss, we return the input tensor as is.
        return input