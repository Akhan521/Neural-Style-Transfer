# NST: Transferring Styles to Content Images

Transform your photos into works of art using Neural Style Transfer! This PyQt6 application allows you to apply artistic styles to your images using a pre-trained VGG19 model.

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