
'''
This file contains the style loss function used in the NST model.

To calculate the style loss, we must first compute the Gram matrix of the feature maps at a certain layer in the VGG19 model.
The Gram matrix is a measure of the correlation between the different feature maps in a layer (i.e., how similar they are to each other).

It's computed by reshaping the feature maps into a 2D matrix and taking the product of the matrix with its transpose.
Each reshaped feature map is a K x N matrix, where K = # of feature maps at a layer L and N = the product of a given feature map's dimensions.
Finally, we normalize the Gram matrix by dividing it by the total number of elements in the matrix.
This normalization ensures that the style loss is not biased towards layers with larger feature maps.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

def gram_matrix(input):

    # Get the dimensions of the input tensor.
    '''
    'batch_size' is the number of images in our batch (1 in our case).
    'num_features' is the number of feature maps in the layer we're observing.
    'height' and 'width' are the dimensions of the feature maps.
    '''
    batch_size, num_features, height, width = input.size() 

    # Reshape the input tensor to have a shape of (num_features, height * width).
    features = input.view(batch_size * num_features, height * width)

    # Compute the Gram matrix by multiplying the reshaped input tensor with its transpose.
    gram_matrix = torch.mm(features, features.t())

    # Normalize the Gram matrix by dividing it by the total number of elements.
    return gram_matrix.div(batch_size * num_features * height * width)

class StyleLoss(nn.Module):

    def __init__(self, target):
        # Initialize the module.
        super(StyleLoss, self).__init__()

        # Compute the Gram matrix of the target feature maps.
        # To prevent PyTorch from tracking the target tensor's gradients, we 'detach' it from the computation graph.
        # This ensures that the target tensor is treated as a constant.
        self.target = gram_matrix(target).detach()

    def forward(self, input):
        # Compute the Gram matrix of the input feature maps.
        G = gram_matrix(input)

        # Compute the MSE loss between the Gram matrices of the input and target feature maps.
        self.loss = F.mse_loss(G, self.target)

        # Since this module is only being used to compute style loss, we return the input tensor as is.
        return input