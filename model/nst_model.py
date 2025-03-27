
'''
This file contains our NST model, which combines the content and style losses to generate a stylized image.

We insert our normalization module at the beginning of the model to preprocess our input images.
Then, we add the content and style loss modules after the convolution layers that produce the feature maps.

We must define our own sequential model since we need to calculate the content and style losses at specific layers in the VGG19 model.
'''

import torch
import torch.nn as nn
import torch.optim as optim

from utils.image_loader import load_image, show_image, show_content_and_style_images
from modules.normalization.norm import Normalization
from modules.loss_functions.content_loss import ContentLoss
from modules.loss_functions.style_loss import StyleLoss

import matplotlib.pyplot as plt


# Desired layers to insert content and style loss modules after.
default_content_layers = ['conv_4']
default_style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_nst_model_and_losses(cnn, norm_mean, norm_std,
                             content_image, style_image,
                             content_layers=default_content_layers,
                             style_layers=default_style_layers):
    
    # Initialize the normalization module.
    normalization = Normalization(norm_mean, norm_std)

    # Initialize lists to store the content and style loss modules.
    content_losses = []
    style_losses = []

    # Define a new sequential model to insert the content and style loss modules into.
    # We'll add the normalization module at the beginning of the model.
    model = nn.Sequential(normalization)

    i = 0 # Counter to keep track of the current convolution layer.
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            # We use the out-of-place version of ReLU to work with the content and style loss modules.
            layer = nn.ReLU(inplace=False)

        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'

        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'

        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')
        
        # Add the current layer to the model.
        model.add_module(name, layer)

        # If we have a convolution layer and it's in our list of content layers, add a content loss module.
        if name in content_layers:
            # Get the target tensor to provide to the content loss module.
            target = model(content_image).detach()

            # Create a content loss module and pass the target tensor to it.
            content_loss_module = ContentLoss(target)

            # Add the content loss module to the model.
            model.add_module(f'content_loss_{i}', content_loss_module)

            # Add the content loss module to the list of content losses.
            content_losses.append(content_loss_module)

        # If we have a convolution layer and it's in our list of style layers, add a style loss module.
        if name in style_layers:
            # Get the target tensor to provide to the style loss module.
            target = model(style_image).detach()

            # Create a style loss module and pass the target tensor to it.
            style_loss_module = StyleLoss(target)

            # Add the style loss module to the model.
            model.add_module(f'style_loss_{i}', style_loss_module)

            # Add the style loss module to the list of style losses.
            style_losses.append(style_loss_module)

    # Trim the model to only include the layers up to the last content or style loss module.
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss)):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses
    
def get_input_optimizer(input_image):
    # Create an Adam optimizer to update the input image.
    # We provide the input image as the parameter to be optimized (requires a gradient).
    optimizer = optim.Adam([input_image], lr=0.02, betas=[0.99, 0.999], eps=1e-1)
    return optimizer

def run_style_transfer(cnn, norm_mean, norm_std,
                       content_image, style_image, input_image,
                       num_steps=1000, style_weight=1e4, content_weight=1e-2):
    
    # To display our intermediate images...
    plt.ion()
    figure, axes = plt.subplots(figsize=(5, 5))
    plt.axis('off')

    # Displaying our input image before optimization.
    show_image(input_image, title='Input Image (Before Optimization)')
    figure.canvas.draw()
    plt.pause(0.5)

    # Get the NST model and the content and style losses.
    model, style_losses, content_losses = get_nst_model_and_losses(cnn, norm_mean, norm_std,
                                                                   content_image, style_image)

    # As we'll only be optimizing the input image, we'll update which parameters require gradients.
    input_image.requires_grad_(True)
    # We put the model in evaluation mode to ensure all layers behave correctly.
    model.eval() 
    model.requires_grad_(False)

    # Create an Adam optimizer to update the input image.
    optimizer = get_input_optimizer(input_image)

    # Run the style transfer process.
    iteration = [0]

    while iteration[0] <= num_steps:

        # Our optimizer requires a closure function to update the input image.
        def closure():
            # Ensure the updated input image values are within the valid range.
            with torch.no_grad():
                input_image.clamp_(0, 1)

            # Reset the gradients of the input image.
            optimizer.zero_grad()

            # Pass the input image through the model to get the content and style losses.
            model(input_image)

            # Initialize the style loss and content loss.
            style_score = 0
            content_score = 0

            # Calculate the style and content losses.
            for style_loss_module in style_losses:
                style_score += style_loss_module.loss
            for content_loss_module in content_losses:
                content_score += content_loss_module.loss

            # Weight the style and content losses.
            style_score *= style_weight
            content_score *= content_weight

            # Compute the gradients of the loss w.r.t. the input image.
            loss = style_score + content_score
            loss.backward()

            # Update the iteration counter.
            iteration[0] += 1

            if iteration[0] % 50 == 0:
                print(f'Iteration {iteration[0]}:\n\t Style Loss: {style_score.item():.4f}\n\t Content Loss: {content_score.item():.4f}\n')
                # Display the input image after every 50 iterations.
                show_image(input_image, title=f'Input Image (Iteration {iteration[0]})')
                figure.canvas.draw()
                plt.pause(0.5)

            return style_score + content_score
        
        # Update the input image using the optimizer.
        optimizer.step(closure)

    # Ensure the updated input image values are within the valid range one last time.
    with torch.no_grad():
        input_image.clamp_(0, 1)

    return input_image