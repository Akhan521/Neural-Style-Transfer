
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
                background-color: #121212;
            }
            /* Main widget background */
            QWidget {
                background-color: #121212;
            }
            /* Labels */
            QLabel {
                color: #FFFFFF;
                font-size: 16px;
                font-weight: bold;
            }
            /* Buttons */
            QPushButton {
                background-color: #1E1E1E;
                color: #FFFFFF;
                border: 1px solid #CE93D8;
                padding: 10px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #CE93D8;
                color: #000000;
            }
            /* Sliders */
            QSlider {
                background-color: #1E1E1E;
                color: #FFFFFF;
            }
            QSlider::groove:horizontal {
                background: #1E1E1E;
                height: 10px;
            }
            QSlider::handle:horizontal {
                background: #CE93D8;
                width: 20px;
                margin: -5px 0;
            }
            QSlider::add-page:horizontal {
                background: #CE93D8;
            }
            QSlider::sub-page:horizontal {
                background: #1E1E1E;
            }
            /* Tooltips */
            QToolTip {
                background-color: #1E1E1E;
                color: #FFFFFF;
                border: 1px solid #CE93D8;
                padding: 5px;
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
        title_label = QLabel("Neural Style Transfer")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #CE93D8;")
        self.main_layout.addWidget(title_label)

        # Our image display area.
        image_layout = QHBoxLayout()
        image_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_layout.setSpacing(20)
        self.content_image_label = QLabel("Content Image")
        self.content_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content_image_label.setFixedSize(300, 300)
        self.style_image_label = QLabel("Style Image")
        self.style_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.style_image_label.setFixedSize(300, 300)
        self.stylized_image_label = QLabel("Stylized Image")
        self.stylized_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stylized_image_label.setFixedSize(300, 300)
        image_layout.addWidget(self.content_image_label)
        image_layout.addWidget(self.style_image_label)
        image_layout.addWidget(self.stylized_image_label)
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
        self.style_weight_slider = QSlider(Qt.Orientation.Horizontal)
        self.style_weight_slider.setRange(0, 1000)
        self.style_weight_slider.setValue(1000)
        self.style_weight_slider.setTickInterval(100)
        self.style_weight_slider.setSingleStep(10)
        self.style_weight_slider.setPageStep(100)
        self.style_weight_slider.setToolTip("Style Weight")
        self.style_weight_slider.setTracking(True)
        self.style_weight_slider.valueChanged.connect(self.update_style_weight_label)
        self.style_weight_label = QLabel("Style Weight: 1000")
        self.content_weight_slider = QSlider(Qt.Orientation.Horizontal)
        self.content_weight_slider.setRange(0, 1000)
        self.content_weight_slider.setValue(100)
        self.content_weight_slider.setTickInterval(100)
        self.content_weight_slider.setSingleStep(10)
        self.content_weight_slider.setPageStep(100)
        self.content_weight_slider.setToolTip("Content Weight")
        self.content_weight_slider.setTracking(True)
        self.content_weight_slider.valueChanged.connect(self.update_content_weight_label)
        self.content_weight_label = QLabel("Content Weight: 100")
        self.num_steps_slider = QSlider(Qt.Orientation.Horizontal)
        self.num_steps_slider.setRange(1, 1000)
        self.num_steps_slider.setValue(500)
        self.num_steps_slider.setTickInterval(100)
        self.num_steps_slider.setSingleStep(10)
        self.num_steps_slider.setPageStep(100)
        self.num_steps_slider.setToolTip("Number of Steps")
        self.num_steps_slider.setTracking(True)
        self.num_steps_slider.valueChanged.connect(self.update_num_steps_label)
        self.num_steps_label = QLabel("Number of Steps: 500")
        param_layout.addWidget(self.style_weight_label)
        param_layout.addWidget(self.style_weight_slider)
        param_layout.addWidget(self.content_weight_label)
        param_layout.addWidget(self.content_weight_slider)
        param_layout.addWidget(self.num_steps_label)
        param_layout.addWidget(self.num_steps_slider)
        self.main_layout.addLayout(param_layout)

    def load_content_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Content Image", "", "Images (*.png *.jpg *.jpeg)")

        if filename:
            # Load the content image and display it.
            self.content_image_path = filename
            content_image = QPixmap(filename).scaled(300, 300, Qt.AspectRatioMode.IgnoreAspectRatio)
            self.content_image_label.setPixmap(content_image)

            # Enable the run button if both images are loaded.
            if self.content_image_path and self.style_image_path:
                self.run_button.setEnabled(True)

    def load_style_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Style Image", "", "Images (*.png *.jpg *.jpeg)")

        if filename:
            # Load the style image and display it.
            self.style_image_path = filename
            style_image = QPixmap(filename).scaled(300, 300, Qt.AspectRatioMode.IgnoreAspectRatio)
            self.style_image_label.setPixmap(style_image)

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

        # Disable the run button and show the progress bar.
        self.run_button.setEnabled(False)
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
        self.stylized_image_label.setPixmap(stylized_image)

        # Enable the run and save buttons and hide the progress bar.
        self.run_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.progress_bar.setVisible(False)

        QMessageBox.information(self, "Success", "Style transfer completed successfully!")

    def on_nst_error(self, error_message):
        # Show an error message if the style transfer fails.
        QMessageBox.critical(self, "Error", f"Style transfer failed: {error_message}")

        # Reset the UI to allow for a new attempt.
        self.reset_ui()

    def save_stylized_image(self):
        if not self.stylized_image_label.pixmap():
            QMessageBox.warning(self, "Error", "No stylized image to save.")
            return

        filename, _ = QFileDialog.getSaveFileName(self, "Save Stylized Image", "", "Images (*.png *.jpg *.jpeg)")

        if filename:
            # Save the stylized image.
            pixmap = self.stylized_image_label.pixmap()
            pixmap.save(filename)
            QMessageBox.information(self, "Success", "Stylized image saved successfully!")

    def reset_ui(self):
        # Reset the image labels and paths.
        self.content_image_label.clear()
        self.style_image_label.clear()
        self.stylized_image_label.clear()
        self.content_image_path = None
        self.style_image_path = None

        # Reset the sliders and labels.
        self.style_weight_slider.setValue(1000)
        self.content_weight_slider.setValue(100)
        self.num_steps_slider.setValue(500)
        self.update_style_weight_label(1000)
        self.update_content_weight_label(100)
        self.update_num_steps_label(500)

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
        


