# NST: Transferring Styles to Content Images

Transform your photos into works of art using Neural Style Transfer! This PyQt6 application allows you to apply artistic styles to your images using a pre-trained VGG19 model.

## Understanding Neural Style Transfer
Given a style reference image and a content reference image, the neural style transfer algorithm is capable of transferring the provided style to our content. This produces a new, stylized image which appears to "merge" the two reference images together. We maintain the style from our style reference image alongside the content from our content reference image within our stylized image.

## Considering Content/Style Loss Tradeoffs
We find that by modifying our content and/or style weights (to consider content/style loss), we can achieve different effects in our stylized image. When we construct our new, stylized image considering only content loss, we are essentially reconstructing our initial content image. On the other hand, if we only consider style loss, we lose content quality but have greater stylization. If we manage to strike a balance with the rightly chosen weights, we can achieve a stylized image that encapsulates the content and style of our reference images. Below we see an example of the effects of content/style loss on our generated image; we see from top to bottom: a stylized image prioritizing content-loss only, prioritizing style-loss only, and prioritizing both losses.
<div style="display: flex; flex-direction: column; align-items: center;">
  <div style="margin-bottom: 20px; text-align: center;">
    <img src="https://github.com/Akhan521/Neural-Style-Transfer/blob/main/screenshots/content_loss_only.png" alt="Content Loss Only" width="75%">
    <p>Considering Only Content Loss</p>
  </div>
  <div style="margin-bottom: 20px; text-align: center;">
    <img src="https://github.com/Akhan521/Neural-Style-Transfer/blob/main/screenshots/style_loss_only.png" alt="Style Loss Only" width="75%">
    <p>Considering Only Style Loss</p>
  </div>
  <div style="text-align: center;">
    <img src="https://github.com/Akhan521/Neural-Style-Transfer/blob/main/screenshots/tiger_as_starry_night.png" alt="Tiger in Starry Night Style" width="75%">
    <p>Considering Both Content and Style Losses</p>
  </div>
</div>

## Features
- Load a content image and a style image.
- Adjust style and content weights using sliders.
- Apply style transfer and view the stylized result.
- Save the stylized image to your computer.
- Dark theme UI for a modern look.

##  🛠 Installation

###  🚀 Download
For users who don't want to install Python, you can download the pre-built executable below:
- [Windows Executable](https://github.com/Akhan521/Neural-Style-Transfer/releases/download/v1.0.0/app.exe)

For a more customizable and hands-on approach, follow the directions below.
### Prerequisites
- Python 3.8 or higher

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Akhan521/Neural-Style-Transfer.git
   cd Neural-Style-Transfer

2. Create a virtual environment + activate it:
    ```bash
    python -m venv venv
    venv\Scripts\activate

3. Install dependencies:
    ```bash
    pip install -r requirements.txt

4. Run the app:
    ```bash
    python app.py

## Usage

1. Launch the app.
2. Click "Load Content Image" to select a content image.
3. Click "Load Style Image" to select a style image.
4. Adjust the style and content weights using the sliders (optional).
5. Adjust the number of steps using the respective slider (optional).
6. Click "Run Style Transfer" to generate the stylized image.
7. Click "Save Stylized Image" to save the result.

## Acknowledgements
* Based on the neural style transfer algorithm by Leon Gatys et al.
* Built with PyQt6 and PyTorch.
