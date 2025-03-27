
'''
A program that demonstrates Neural Style Transfer using PyTorch.
We'll further develop this program using Flask and deploy it on Google Cloud Platform.
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image
import matplotlib.pyplot as plt

import copy 

def load_image(image_path):
    # Load the image using PIL.
    image = Image.open(image_path)

    # Transform the image and add a dummy batch dimension to fit the model's input dimensions.
    image = loader(image).unsqueeze(0)

    return image.to(device, torch.float)

def show_image(tensor, title=None):
    # Clone the tensor to avoid modifying the original tensor.
    image = tensor.cpu().clone()

    # Remove the dummy batch dimension that we added earlier and convert the tensor to a PIL image.
    image = image.squeeze(0)
    image = unloader(image)

    # Display the image.
    plt.imshow(image)
    if title:
        plt.title(title)

    # To ensure that the plot is updated, we need to pause for a bit.
    plt.pause(0.001) 



if __name__ == '__main__':
    
    # Set up the device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    # Specify the max resolution for our images.
    max_res = 512 if torch.cuda.is_available() else 128

    # Define a transformer to resize the image and convert it to a tensor.
    loader = transforms.Compose([
        transforms.Resize(max_res),
        transforms.ToTensor()
    ])

    # Load the content and style images.
    content_image = load_image("images/content_images/dancing.jpg")
    style_image = load_image("images/style_images/picasso.jpg")

    # To display our images from tensors, we need to convert them back to PIL images.
    unloader = transforms.ToPILImage()

    # Display the content and style images:
    plt.figure(figsize=(10, 5))

    # Display the content image on the left.
    plt.subplot(1, 2, 1)
    show_image(content_image, title='Content Image')

    # Display the style image on the right.
    plt.subplot(1, 2, 2)
    show_image(style_image, title='Style Image')

    plt.show()