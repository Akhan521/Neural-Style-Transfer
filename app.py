
'''
A PyQt6 app that demonstrates Neural Style Transfer using PyTorch.
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
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSlider, QLineEdit, QFileDialog, QMessageBox,
                             QProgressBar, QToolTip)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage

from werkzeug.utils import secure_filename

# A worker thread to run the style transfer in the background.
class NSTWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, cnn, content_image_path, style_image_path, num_steps, style_weight, content_weight):
        super().__init__()
        self.cnn = cnn
        self.content_image_path = content_image_path
        self.style_image_path = style_image_path
        self.num_steps = num_steps
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.stylized_image_path = 'images/stylized_images/'
    
    def run(self):
        try:
            # Load the content and style images using our image loader.
            content_image = load_image(self.content_image_path)
            style_image = load_image(self.style_image_path)
            input_image = content_image.clone()

            # Run the style transfer algorithm.
            stylized_image = run_style_transfer(
                self.cnn, cnn_norm_mean, cnn_norm_std,
                content_image, style_image, input_image,
                num_steps=self.num_steps, style_weight=self.style_weight, content_weight=self.content_weight,
                progress_callback=self.update_progress
            )

            # Convert our stylized image tensor to a PIL image.
            stylized_image = tensor_to_image(stylized_image)

            # Prepare the stylized image for saving.
            adjusted_content_filename = self.content_image_path.rsplit('.', 1)[0].rsplit('/', 1)[-1]
            adjusted_content_filename = secure_filename(adjusted_content_filename)
            adjusted_style_filename = self.style_image_path.rsplit('/', 1)[-1]
            adjusted_style_filename = secure_filename(adjusted_style_filename)
            stylized_image_filename = f'{adjusted_content_filename}_as_{adjusted_style_filename}'
            stylized_image_path = os.path.join(self.stylized_image_path, stylized_image_filename)
            stylized_image.save(stylized_image_path)

            # Emit the finished signal with the path to the stylized image.
            self.finished.emit(stylized_image_path)

        except Exception as e:
            # Emit the error signal with the error message.
            self.error.emit(str(e))

    def update_progress(self, step, total_steps):
        # Calculate the progress percentage.
        progress_percentage = int((step / total_steps) * 100)
        self.progress.emit(progress_percentage)


class NSTWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Style Transfer")
        self.setGeometry(100, 100, 1000, 800)
        self.setMinimumSize(800, 600)

        # Set up our torch device.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self.device)

        # Load the VGG19 model and set it to evaluation mode.
        self.cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

        # Paths to our images.
        self.content_image_path = None
        self.style_image_path = None
        
        # Our NST worker thread.
        self.worker = None

        # Init. our UI.
        self.init_ui()

        self.apply_dark_theme()

    def apply_dark_theme(self):
        dark_theme = '''
            /* Main window background */
            QMainWindow {
                background-color: #1C2526;
            }

            /* Labels (for text and images) */
            QLabel {
                color: #B0B0B0;
                font-size: 14px;
                font-family: "Arial";
                background-color: #2E3A3B;
                border: none;
                border-radius: 10px;
                padding: 5px;
            }

            /* Title label */
            QLabel#titleLabel {
                font-size: 22px;
                font-weight: normal;
                color: #D3D3D3; /* Light gray for subtle contrast */
                background-color: #2E3A3B;
                border-radius: 5px;
                padding: 10px;
                text-align: center;
                text-transform: none;
                font-family: "Arial", sans-serif;
            }

            /* Buttons */
            QPushButton {
                background-color: #2E3A3B;
                color: #B0B0B0;
                font-size: 14px;
                font-family: "Arial";
                border: none;
                border-radius: 10px;
                padding: 10px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #AB47BC; /* Purple background on hover */
                color: #FFFFFF; /* White text on hover */
            }
            QPushButton:pressed {
                background-color: #7B1FA2;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }

            /* Sliders */
            QSlider {
                margin: 10px;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #2E3A3B;
                border: none;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4682B4;
                border: none;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #AB47BC; /* No transition */
            }

            /* Progress Bar */
            QProgressBar {
                background-color: #2E3A3B;
                border: none;
                border-radius: 5px;
                text-align: center;
                color: #FFFFFF; /* White text for readability */
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 5px;
            }

            /* Tooltips */
            QToolTip {
                background-color: #2E3A3B;
                color: #B0B0B0;
                border: none;
                border-radius: 5px;
                padding: 5px;
            }

            /* Message boxes */
            QMessageBox {
                background-color: #1C2526;
                color: #B0B0B0;
            }
            QMessageBox QLabel {
                color: #B0B0B0;
                font-size: 14px;
                font-family: "Arial";
                background: transparent;
                padding: 5px;
            }
            QMessageBox QPushButton {
                background-color: #2E3A3B;
                color: #B0B0B0;
                border: none;
                border-radius: 5px;
                padding: 5px;
            }
            QMessageBox QPushButton:hover {
                background-color: #AB47BC;
            }
        '''
        self.setStyleSheet(dark_theme)

    def init_ui(self):
        
        # Our main widget.
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.main_layout = QVBoxLayout(main_widget)
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.setSpacing(20)

        # Our title label.
        title_label = QLabel("Style Transfer : Transform Your Photos Into Works Of Art")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(title_label)

        # Our image display area.
        image_layout = QHBoxLayout()
        image_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_layout.setSpacing(20)

        # Our content layout.
        content_layout = QVBoxLayout()
        content_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.setSpacing(10)
        self.content_image = QLabel()
        self.content_image_label = QLabel("CONTENT")
        self.content_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content_image.setFixedSize(300, 300)
        content_layout.addWidget(self.content_image)
        content_layout.addWidget(self.content_image_label)
        image_layout.addLayout(content_layout)

        # Our style layout.
        style_layout = QVBoxLayout()
        style_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        style_layout.setSpacing(10)
        self.style_image = QLabel()
        self.style_image_label = QLabel("STYLE")
        self.style_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.style_image.setFixedSize(300, 300)
        style_layout.addWidget(self.style_image)
        style_layout.addWidget(self.style_image_label)
        image_layout.addLayout(style_layout)

        # Our stylized layout.
        stylized_layout = QVBoxLayout()
        stylized_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stylized_layout.setSpacing(10)
        self.stylized_image = QLabel()
        self.stylized_image_label = QLabel("STYLIZED")
        self.stylized_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stylized_image.setFixedSize(300, 300)
        stylized_layout.addWidget(self.stylized_image)
        stylized_layout.addWidget(self.stylized_image_label)
        image_layout.addLayout(stylized_layout)
        self.main_layout.addLayout(image_layout)

        # Our content + style image buttons.
        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        button_layout.setSpacing(20)
        self.load_content_button = QPushButton("Load Content Image")
        self.load_content_button.clicked.connect(self.load_content_image)
        self.load_style_button = QPushButton("Load Style Image")
        self.load_style_button.clicked.connect(self.load_style_image)

        # Add our buttons to the layout.
        button_layout.addWidget(self.load_content_button)
        button_layout.addWidget(self.load_style_button)
        self.main_layout.addLayout(button_layout)

        # Our parameter sliders.
        self.init_param_sliders()

        # Our action buttons: Run and Save.
        action_layout = QHBoxLayout()
        action_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        action_layout.setSpacing(20)
        self.run_button = QPushButton("Run Style Transfer")
        self.run_button.clicked.connect(self.run_nst)
        self.run_button.setEnabled(False)
        self.run_button.setToolTip("Run Neural Style Transfer")
        self.run_button.setFixedSize(200, 50)
        self.run_button.setCursor(Qt.CursorShape.PointingHandCursor)

        # Our save button.
        self.save_button = QPushButton("Save Stylized Image")
        self.save_button.setEnabled(False)
        self.save_button.setToolTip("Save the stylized image")
        self.save_button.setFixedSize(200, 50)
        self.save_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_button.clicked.connect(self.save_stylized_image)

        # Our reset button.
        self.reset_button = QPushButton("Reset")
        self.reset_button.setToolTip("Reset images and parameters")
        self.reset_button.setFixedSize(200, 50)
        self.reset_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reset_button.clicked.connect(self.reset_ui)

        # Add our action buttons to the layout.
        action_layout.addWidget(self.run_button)
        action_layout.addWidget(self.save_button)
        action_layout.addWidget(self.reset_button)
        self.main_layout.addLayout(action_layout)

        # A progress bar to show the progress of the style transfer.
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.main_layout.addWidget(self.progress_bar)

    def init_param_sliders(self):
        # Our parameter sliders.
        param_layout = QVBoxLayout()
        param_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        param_layout.setSpacing(10)

        # Our style weight slider.
        style_layout = QHBoxLayout()
        style_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        style_layout.setSpacing(10)
        self.style_weight_label = QLabel("Style Weight: 1000000")
        self.style_weight_label.setFixedSize(200, 30)
        self.style_weight_slider = QSlider(Qt.Orientation.Horizontal)
        self.style_weight_slider.setRange(0, 1000000)
        self.style_weight_slider.setValue(1000000)
        self.style_weight_slider.setTickInterval(100)
        self.style_weight_slider.setSingleStep(10)
        self.style_weight_slider.setPageStep(100)
        self.style_weight_slider.setToolTip("Style Weight")
        self.style_weight_slider.setTracking(True)
        self.style_weight_slider.valueChanged.connect(self.update_style_weight_label)
        self.style_weight_slider.setCursor(Qt.CursorShape.PointingHandCursor)
        style_layout.addWidget(self.style_weight_label)
        style_layout.addWidget(self.style_weight_slider)
        param_layout.addLayout(style_layout)

        # Our content weight slider.
        content_layout = QHBoxLayout()
        content_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.setSpacing(10)
        self.content_weight_label = QLabel("Content Weight: 1")
        self.content_weight_label.setFixedSize(200, 30)
        self.content_weight_slider = QSlider(Qt.Orientation.Horizontal)
        self.content_weight_slider.setRange(-1000, 1000)
        self.content_weight_slider.setValue(1)
        self.content_weight_slider.setTickInterval(100)
        self.content_weight_slider.setSingleStep(10)
        self.content_weight_slider.setPageStep(100)
        self.content_weight_slider.setToolTip("Content Weight")
        self.content_weight_slider.setTracking(True)
        self.content_weight_slider.valueChanged.connect(self.update_content_weight_label)
        self.content_weight_slider.setCursor(Qt.CursorShape.PointingHandCursor)
        content_layout.addWidget(self.content_weight_label)
        content_layout.addWidget(self.content_weight_slider)
        param_layout.addLayout(content_layout)

        # Our number of steps slider.
        num_steps_layout = QHBoxLayout()
        num_steps_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        num_steps_layout.setSpacing(10)
        self.num_steps_label = QLabel("Number of Steps: 300")
        self.num_steps_label.setFixedSize(200, 30)
        self.num_steps_slider = QSlider(Qt.Orientation.Horizontal)
        self.num_steps_slider.setRange(1, 500)
        self.num_steps_slider.setValue(300)
        self.num_steps_slider.setTickInterval(100)
        self.num_steps_slider.setSingleStep(10)
        self.num_steps_slider.setPageStep(100)
        self.num_steps_slider.setToolTip("Number of Steps")
        self.num_steps_slider.setTracking(True)
        self.num_steps_slider.valueChanged.connect(self.update_num_steps_label)
        self.num_steps_slider.setCursor(Qt.CursorShape.PointingHandCursor)
        num_steps_layout.addWidget(self.num_steps_label)
        num_steps_layout.addWidget(self.num_steps_slider)
        param_layout.addLayout(num_steps_layout)

        # Add our params. to the main layout.
        self.main_layout.addLayout(param_layout)

    def load_content_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Content Image", "", "Images (*.png *.jpg *.jpeg)")

        if filename:
            # Load the content image and display it.
            self.content_image_path = filename
            content_image = QPixmap(filename).scaled(300, 300, Qt.AspectRatioMode.IgnoreAspectRatio)
            self.content_image.setPixmap(content_image)

            # Enable the run button if both images are loaded.
            if self.content_image_path and self.style_image_path:
                self.run_button.setEnabled(True)

    def load_style_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Style Image", "", "Images (*.png *.jpg *.jpeg)")

        if filename:
            # Load the style image and display it.
            self.style_image_path = filename
            style_image = QPixmap(filename).scaled(300, 300, Qt.AspectRatioMode.IgnoreAspectRatio)
            self.style_image.setPixmap(style_image)

        # Enable the run button if both images are loaded.
        if self.content_image_path and self.style_image_path:
            self.run_button.setEnabled(True)

    def run_nst(self):
        # If either image is not loaded, show an error message.
        if not self.content_image_path or not self.style_image_path:
            QMessageBox.warning(self, "Error", "Please load both content and style images.")

        # Get the parameters from the sliders.
        style_weight = self.style_weight_slider.value()
        content_weight = self.content_weight_slider.value()
        num_steps = self.num_steps_slider.value()

        # Disable all buttons and show the progress bar.
        self.load_content_button.setEnabled(False)
        self.load_style_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Create a new worker thread to run the style transfer.
        self.worker = NSTWorker(
            self.cnn, self.content_image_path, self.style_image_path,
            num_steps, style_weight, content_weight
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_nst_finished)
        self.worker.error.connect(self.on_nst_error)
        self.worker.start()

    def update_progress(self, progress):
        # Update the progress bar with the current progress.
        self.progress_bar.setValue(progress)

    def on_nst_finished(self, stylized_image_path):
        # Load the stylized image and display it.
        stylized_image = QPixmap(stylized_image_path).scaled(300, 300, Qt.AspectRatioMode.IgnoreAspectRatio)
        self.stylized_image.setPixmap(stylized_image)

        # Enable the all buttons and hide the progress bar.
        self.load_content_button.setEnabled(True)
        self.load_style_button.setEnabled(True)
        self.run_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.progress_bar.setVisible(False)

        QMessageBox.information(self, "Success", "Style transfer completed successfully!")

    def on_nst_error(self, error_message):
        # Show an error message if the style transfer fails.
        QMessageBox.critical(self, "Error", f"Style transfer failed: {error_message}")

        # Reset the UI to allow for a new attempt.
        self.reset_ui()

    def save_stylized_image(self):
        if not self.stylized_image.pixmap():
            QMessageBox.warning(self, "Error", "No stylized image to save.")
            return

        filename, _ = QFileDialog.getSaveFileName(self, "Save Stylized Image", "", "Images (*.png *.jpg *.jpeg)")

        if filename:
            # Save the stylized image.
            pixmap = self.stylized_image.pixmap()
            pixmap.save(filename)
            QMessageBox.information(self, "Success", "Stylized image saved successfully!")

    def reset_ui(self):
        # Reset the image labels and paths.
        self.content_image.clear()
        self.style_image.clear()
        self.stylized_image.clear()
        self.content_image_path = None
        self.style_image_path = None

        # Reset the sliders and labels.
        self.style_weight_slider.setValue(1000000)
        self.content_weight_slider.setValue(1)
        self.num_steps_slider.setValue(300)
        self.update_style_weight_label(1000000)
        self.update_content_weight_label(1)
        self.update_num_steps_label(300)

        # Enable the load buttons + reset button.
        self.load_content_button.setEnabled(True)
        self.load_style_button.setEnabled(True)
        self.reset_button.setEnabled(True)

        # Disable the run and save buttons.
        self.run_button.setEnabled(False)
        self.save_button.setEnabled(False)

        # Hide the progress bar.
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)

    def update_style_weight_label(self, value):
        self.style_weight_label.setText(f"Style Weight: {value}")
        self.style_weight_slider.setToolTip(f"Style Weight: {value}")

    def update_content_weight_label(self, value):
        self.content_weight_label.setText(f"Content Weight: {value}")
        self.content_weight_slider.setToolTip(f"Content Weight: {value}")

    def update_num_steps_label(self, value):
        self.num_steps_label.setText(f"Number of Steps: {value}")
        self.num_steps_slider.setToolTip(f"Number of Steps: {value}")

if __name__ == "__main__":
    app = QApplication([])
    window = NSTWindow()
    window.showMaximized()
    app.exec()
    app.quit()
        


