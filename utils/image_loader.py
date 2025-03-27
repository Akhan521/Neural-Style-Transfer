
import torch
import torchvision.transforms as transforms

from PIL import Image
import matplotlib.pyplot as plt

# Set up the device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Specify the max resolution for our images.
max_res = 512 if torch.cuda.is_available() else 128

# Define a transformer to resize the image and convert it to a tensor.
loader = transforms.Compose([
    transforms.Resize(max_res),
    transforms.ToTensor()
])

# To display our images from tensors, we need to convert them back to PIL images.
unloader = transforms.ToPILImage()

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

def show_content_and_style_images(content_image, style_image):
    # Display the content and style images.
    plt.figure(figsize=(10, 5))

    # Display the content image on the left.
    plt.subplot(1, 2, 1)
    show_image(content_image, title='Content Image')

    # Display the style image on the right.
    plt.subplot(1, 2, 2)
    show_image(style_image, title='Style Image')

    # Pause for a few seconds to display the images.
    plt.pause(5)
    plt.close()
