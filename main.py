
'''
A program that demonstrates Neural Style Transfer using PyTorch.
We'll further develop this program using Flask and deploy it on Google Cloud Platform.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights

from utils.image_loader import load_image, show_image, show_content_and_style_images
from modules.normalization.norm import cnn_norm_mean, cnn_norm_std
from model.nst_model import run_style_transfer

from PIL import Image
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    # Set up the device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    # Load the content and style images.
    content_image = load_image("images/content_images/dancing.jpg")
    style_image = load_image("images/style_images/picasso.jpg")

    # Display the content and style images.
    plt.ion()
    show_content_and_style_images(content_image, style_image)

    # Load the VGG19 model and set it to evaluation mode.
    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

    # Set the input image.
    input_image = content_image.clone()

    # Run the style transfer algorithm.
    stylized_image = run_style_transfer(cnn, cnn_norm_mean, cnn_norm_std,
                                        content_image, style_image, input_image)
    
