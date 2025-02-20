# create a pyside6 application with widget that filters noise from an image
#make most of the functions from scratch
# use numpy to filter noise from the image and display the filtered image
# types of filters: average, median, gaussian filters

import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QWidget, QFileDialog, \
    QComboBox, QSlider
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
import cv2

class FilterNoiseWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.modified_image = self.image
        self.setWindowTitle("Filter Noise")
        self.layout = QVBoxLayout()
        self.image_label = QLabel()
        
        # temporary fix for size issues
        self.image_label.setMaximumSize(600, 400)
        
        self.layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        # add button to load image
        self.load_image_button = QPushButton("Load Image")
        self.load_image_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_image_button)
        # add dropdown menu to select the type of filter (average, median, gaussian)
        self.filter_type_menu = QComboBox()
        self.filter_type_menu.addItems(["Average", "Median", "Gaussian"])
        self.layout.addWidget(self.filter_type_menu)
        # add sliders to adjust the filter size
        self.filter_size_slider = QSlider(Qt.Horizontal)
        self.filter_size_slider.setMinimum(0)
        self.filter_size_slider.setMaximum(10)
        self.filter_size_slider.setValue(3)
        self.layout.addWidget(self.filter_size_slider)
        # add button to filter noise
        self.filter_noise_button = QPushButton("Filter Noise")
        self.filter_noise_button.clicked.connect(self.filter_noise)
        self.layout.addWidget(self.filter_noise_button)
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
        
        # temporary fix for size issues
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)  

    def filter_noise(self):
        if self.image is None:
            return
        filter_type = self.filter_type_menu.currentText()
        filter_size = self.filter_size_slider.value()
        # use filters from scratch (implement the filters using numpy)
        if filter_type == "Average":
            self.modified_image = self.average_filter(self.image, filter_size)
        elif filter_type == "Median":
            self.modified_image = self.median_filter(self.image, filter_size)
        elif filter_type == "Gaussian":
            self.modified_image = self.gaussian_filter(self.image, filter_size)
        self.show_image()

    def average_filter(self, image, filter_size):
        kernel = np.ones((filter_size, filter_size), np.float32) / (filter_size * filter_size)
        padded_image = np.pad(image,
                              ((filter_size // 2, filter_size // 2), (filter_size // 2, filter_size // 2), (0, 0)),
                              mode='constant')
        filtered_image = np.zeros_like(image)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    filtered_image[i, j, k] = np.sum(padded_image[i:i + filter_size, j:j + filter_size, k] * kernel)

        return filtered_image

    def median_filter(self, image, filter_size):
        padded_image = np.pad(image,
                              ((filter_size // 2, filter_size // 2), (filter_size // 2, filter_size // 2), (0, 0)),
                              mode='constant')
        filtered_image = np.zeros_like(image)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    filtered_image[i, j, k] = np.median(padded_image[i:i + filter_size, j:j + filter_size, k])

        return filtered_image

    def gaussian_filter(self, image, filter_size, sigma=1.0):
        # Ensure filter size is odd
        if filter_size % 2 == 0:
            filter_size += 1

        ax = np.linspace(-(filter_size - 1) / 2., (filter_size - 1) / 2., filter_size)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
        kernel = np.outer(gauss, gauss)
        kernel /= np.sum(kernel)

        padded_image = np.pad(image,
                              ((filter_size // 2, filter_size // 2), (filter_size // 2, filter_size // 2), (0, 0)),
                              mode='constant')
        filtered_image = np.zeros_like(image)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    filtered_image[i, j, k] = np.sum(padded_image[i:i + filter_size, j:j + filter_size, k] * kernel)

        return filtered_image

# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Filter Noise")
#         self.setCentralWidget(FilterNoiseWidget())

# app = QApplication([])
# window = MainWindow()
# window.show()
# sys.exit(app.exec())


