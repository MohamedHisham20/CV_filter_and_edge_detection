# create a pyside6 application that adds noise to an image
#make most of the functions from scratch and use numpy to add noise to the image

import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QWidget, QFileDialog, \
    QComboBox, QSlider
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
import cv2

class AddNoiseWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.modified_image = self.image
        self.setWindowTitle("Add Noise")
        self.layout = QVBoxLayout()
        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)
        # add button to load image
        self.load_image_button = QPushButton("Load Image")
        self.load_image_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_image_button)
        # add dropdown menu to select the type of noise (gaussian, salt and pepper, uniform)
        self.noise_type_menu = QComboBox()
        self.noise_type_menu.addItems(["Gaussian", "Salt and Pepper", "Uniform"])
        self.layout.addWidget(self.noise_type_menu)
        # add sliders to adjust the noise level
        self.noise_level_slider = QSlider(Qt.Horizontal)
        self.noise_level_slider.setMinimum(0)
        self.noise_level_slider.setMaximum(100)
        self.noise_level_slider.setValue(50)
        self.layout.addWidget(self.noise_level_slider)
        # add button to add noise
        self.add_noise_button = QPushButton("Add Noise")
        self.add_noise_button.clicked.connect(self.add_noise)
        self.layout.addWidget(self.add_noise_button)
        # add button to save image
        self.save_image_button = QPushButton("Save Image")
        self.save_image_button.clicked.connect(self.save_image)
        self.layout.addWidget(self.save_image_button)
        self.setLayout(self.layout)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.image = cv2.imread(file_name)
            self.modified_image = self.image
            self.show_image()

    def show_image(self):
        if len(self.modified_image.shape) == 3 and self.modified_image.shape[2] == 3:
            self.modified_image = cv2.cvtColor(self.modified_image, cv2.COLOR_BGR2RGB)
        h, w, ch = self.modified_image.shape
        bytes_per_line = ch * w
        qimage = QImage(self.modified_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap)

    def save_image(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            # Convert the image back to BGR format before saving
            image_to_save = cv2.cvtColor(self.modified_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_name, image_to_save)

    def add_noise(self):
        if self.image is None:
            return
        noise = self.noise_type_menu.currentText()
        level = self.noise_level_slider.value() / 100
        if noise == "Gaussian":
            self.modified_image = self.add_gaussian_noise(self.image, level)
        elif noise == "Salt and Pepper":
            self.modified_image = self.add_salt_and_pepper_noise(self.image, level)
        elif noise == "Uniform":
            self.modified_image = self.add_uniform_noise(self.image, level)
        self.show_image()

    def add_salt_and_pepper_noise(self, image, level):
        noisy_image = np.copy(image)
        salt = np.random.rand(*image.shape[:2]) < level / 2
        pepper = np.random.rand(*image.shape[:2]) < level / 2
        noisy_image[salt] = 255
        noisy_image[pepper & ~salt] = 0
        return noisy_image

    def add_gaussian_noise(self, image, level):
        noise = np.random.normal(0, 1, image.shape) * 255 * level
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image

    # def add_salt_and_pepper_noise(self, image, level):
    #     noisy_image = np.copy(image)
    #     salt = np.random.rand(*image.shape[:2]) < level / 2
    #     pepper = np.random.rand(*image.shape[:2]) < level / 2
    #     noisy_image[salt] = 255
    #     noisy_image[pepper] = 0
    #     return noisy_image

    def add_uniform_noise(self, image, level):
        noise = np.random.uniform(-255 * level, 255 * level, image.shape)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Add Noise")
        self.setCentralWidget(AddNoiseWidget())

app = QApplication([])
window = MainWindow()
window.show()
sys.exit(app.exec())