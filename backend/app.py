
'''
A flask app that demonstrates Neural Style Transfer using PyTorch.
'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights

from utils.image_loader import load_image, tensor_to_image
from modules.normalization.norm import cnn_norm_mean, cnn_norm_std
from model.nst_model import run_style_transfer

from PIL import Image
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename

# Init. our Flask app.
app = Flask(__name__)

# Set the upload folder and allowed extensions.
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Ensure the upload folder exists.
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
# Set up our device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# Load the VGG19 model and set it to evaluation mode.
cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

# Check if the provided filename is allowed (i.e., has an allowed extension).
def is_allowed_file(filename):
    # We split our filename at the '.' only once
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Our index route.
@app.route('/')
def index():
    return render_template('index.html')

# Our route for processing our images and running the style transfer algorithm.
@app.route('/neural-style-transfer', methods=['POST'])
def process_and_run_nst():
    # If the request is missing either the content or style image, return an error.
    if 'content_image' not in request.files or 'style_image' not in request.files:
        return jsonify({'error': 'You must provide both a content and style image.'}), 400

    content_image = request.files['content_image']
    style_image = request.files['style_image']

    # Verify that the files are allowed.
    if content_image == '' or style_image == '':
        return jsonify({'error': 'You must select files for both the content and style images.'}), 400
    
    if not is_allowed_file(content_image.filename) or not is_allowed_file(style_image.filename):
        return jsonify({'error': 'Only PNG, JPG, and JPEG files are allowed.'}), 400

    # Save the content and style images to the uploads folder.
    content_image_filename = secure_filename(content_image.filename)
    style_image_filename = secure_filename(style_image.filename)
    content_image_path = os.path.join(app.config['UPLOAD_FOLDER'], content_image_filename)
    style_image_path = os.path.join(app.config['UPLOAD_FOLDER'], style_image_filename)
    content_image.save(content_image_path)
    style_image.save(style_image_path)

    # Load the content, style, and input images.
    content_image = load_image(content_image_path)
    style_image = load_image(style_image_path)
    input_image = content_image.clone()

    # Run the style transfer algorithm.
    stylized_image = run_style_transfer(
        cnn, cnn_norm_mean, cnn_norm_std,
        content_image, style_image, input_image,
        num_steps=500, style_weight=1e6, content_weight=1e-2
    )

    # Prepare the stylized image for saving.
    stylized_image_filename = f'stylized_{content_image_filename}_as_{style_image_filename}'
    stylized_image_path = os.path.join(app.config['UPLOAD_FOLDER'], stylized_image_filename)
    # Convert our stylized image tensor to a PIL image.
    stylized_image = tensor_to_image(stylized_image)
    # Save the image.
    stylized_image.save(stylized_image_path)

    # Return our image paths to display in the frontend.
    return jsonify({
        'content_image': f'/uploads/{content_image_filename}',
        'style_image': f'/uploads/{style_image_filename}',
        'stylized_image': f'/uploads/{stylized_image_filename}'
    })

# Our route to serve/get the uploaded files.
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

