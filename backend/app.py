
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
from flask_cors import CORS
from werkzeug.utils import secure_filename
import firebase_admin
from firebase_admin import credentials, storage
import config
import tempfile
import json

# Init. our Flask app.
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Set up Firebase Storage bucket.
bucket = storage.bucket()

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

    # Upload the content image to Firebase Storage.
    content_image_filename = secure_filename(content_image.filename)
    content_blob = bucket.blob(f'content_images/{content_image_filename}')
    content_blob.upload_from_file(content_image)
    content_blob.make_public()  
    content_image_url = content_blob.public_url

    # Upload the style image to Firebase Storage.
    style_image_filename = secure_filename(style_image.filename)
    style_blob = bucket.blob(f'style_images/{style_image_filename}')
    style_blob.upload_from_file(style_image)
    style_blob.make_public()
    style_image_url = style_blob.public_url

    # Reset the file pointers to the beginning of the files.
    content_image.seek(0)
    style_image.seek(0)

    # Create temporary files for the content and style images (for local processing).
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{content_image_filename.rsplit(".", 1)[1]}') as temp_file:
        content_image.save(temp_file.name)
        content_image_path = temp_file.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{style_image_filename.rsplit(".", 1)[1]}') as temp_file:
        style_image.save(temp_file.name)
        style_image_path = temp_file.name
    
    try:
        # Load the content and style images using our image loader.
        content_image = load_image(content_image_path)
        style_image = load_image(style_image_path)
    
        # Create our input image as a clone of the content image.
        input_image = content_image.clone()

        # Run the style transfer algorithm.
        stylized_image = run_style_transfer(
            cnn, cnn_norm_mean, cnn_norm_std,
            content_image, style_image, input_image,
            num_steps=100, style_weight=1e4, content_weight=1e-2
        )

        # Prepare the stylized image for saving.
        adjusted_content_img_filename = content_image_filename.rsplit('.', 1)[0].rsplit('/', 1)[-1]
        stylized_image_filename = f'stylized_{adjusted_content_img_filename}_as_{style_image_filename}'

        # Convert our stylized image tensor to a PIL image.
        stylized_image = tensor_to_image(stylized_image)
        
        # Save the stylized image to a temporary location for uploading to GCS.
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            stylized_image.save(temp_file.name)
            temp_file_path = temp_file.name

        # Upload the stylized image to Firebase Storage.
        stylized_blob = bucket.blob(f'stylized_images/{stylized_image_filename}')
        stylized_blob.upload_from_filename(temp_file_path)
        stylized_blob.make_public()
        stylized_image_url = stylized_blob.public_url

        # Remove the temporary files after uploading.
        os.remove(content_image_path)
        os.remove(style_image_path)
        os.remove(temp_file_path)

        # Return the public URLs of the images.
        return jsonify({
            'content_image': content_image_url,
            'style_image': style_image_url,
            'stylized_image': stylized_image_url
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Our route to serve/get the uploaded files.
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Use PORT env variable for Render.
    app.run(host="0.0.0.0", port=port)


