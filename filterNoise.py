import numpy as np
from PySide6.QtWidgets import QLabel, QVBoxLayout, QPushButton, QWidget, QFileDialog, QComboBox, \
    QSlider, QHBoxLayout
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
import cv2



class FilterNoiseWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.modified_image = None
        self.original_pixmap = None
        self.setWindowTitle("Filter Noise")

        # Main layout
        self.layout = QVBoxLayout()

        # Image display
        self.image_label = QLabel()
        self.image_label.setMinimumSize(600, 400)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        # Load image button
        self.load_image_button = QPushButton("Load Image")
        self.load_image_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_image_button)

        # Filter type selector
        filter_type_layout = QHBoxLayout()
        filter_type_layout.addWidget(QLabel("Filter Type:"))
        self.filter_type_menu = QComboBox()
        self.filter_type_menu.addItems(["Average", "Median", "Gaussian"])
        filter_type_layout.addWidget(self.filter_type_menu)
        self.layout.addLayout(filter_type_layout)

        # Filter size slider
        filter_size_layout = QHBoxLayout()
        filter_size_layout.addWidget(QLabel("Filter Size:"))
        self.filter_size_slider = QSlider(Qt.Horizontal)
        self.filter_size_slider.setMinimum(1)  # Start from 1
        self.filter_size_slider.setMaximum(5)  # Limit to reasonable sizes
        self.filter_size_slider.setValue(1)
        self.filter_size_label = QLabel("3")  # Starting value (2*1+1)
        self.filter_size_slider.valueChanged.connect(self.update_filter_size_label)
        filter_size_layout.addWidget(self.filter_size_slider)
        filter_size_layout.addWidget(self.filter_size_label)
        self.layout.addLayout(filter_size_layout)

        # For Gaussian filter - sigma value
        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel("Sigma (Gaussian only):"))
        self.sigma_slider = QSlider(Qt.Horizontal)
        self.sigma_slider.setMinimum(10)  # 0.1 after division
        self.sigma_slider.setMaximum(50)  # 5.0 after division
        self.sigma_slider.setValue(10)
        self.sigma_label = QLabel("1.0")
        self.sigma_slider.valueChanged.connect(self.update_sigma_label)
        sigma_layout.addWidget(self.sigma_slider)
        sigma_layout.addWidget(self.sigma_label)
        self.layout.addLayout(sigma_layout)

        # Filter button and reset button
        button_layout = QHBoxLayout()
        self.filter_noise_button = QPushButton("Apply Filter")
        self.filter_noise_button.clicked.connect(self.filter_noise)
        button_layout.addWidget(self.filter_noise_button)

        self.reset_button = QPushButton("Reset Image")
        self.reset_button.clicked.connect(self.reset_image)
        button_layout.addWidget(self.reset_button)
        self.layout.addLayout(button_layout)

        self.setLayout(self.layout)

        # Initialize UI state
        self.update_filter_size_label(self.filter_size_slider.value())
        self.update_sigma_label(self.sigma_slider.value())

    def update_filter_size_label(self, value):
        # Convert slider value to actual filter size (always odd)
        filter_size = 2 * value + 1
        self.filter_size_label.setText(str(filter_size))

    def update_sigma_label(self, value):
        # Convert slider value to sigma (divide by 10 for decimal)
        sigma = value / 10.0
        self.sigma_label.setText(str(sigma))

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.image = cv2.imread(file_name)
            if self.image is None:
                return

            # Convert to RGB for display
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.modified_image = self.image.copy()
            self.show_image()

    def show_image(self):
        if self.modified_image is None:
            return

        h, w, ch = self.modified_image.shape
        bytes_per_line = ch * w
        qimage = QImage(self.modified_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        # Store original pixmap for reset
        if self.original_pixmap is None:
            self.original_pixmap = pixmap

        # Scale pixmap to fit in label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(self.image_label.size(),
                                      Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation)

        self.image_label.setPixmap(scaled_pixmap)

    def reset_image(self):
        if self.image is None:
            return

        self.modified_image = self.image.copy()
        self.show_image()

    def filter_noise(self):
        if self.image is None:
            return

        filter_type = self.filter_type_menu.currentText()
        filter_size = 2 * self.filter_size_slider.value() + 1  # Convert to odd number
        sigma = self.sigma_slider.value() / 10.0

        if filter_type == "Average":
            self.modified_image = self.average_filter(self.modified_image, filter_size)
        elif filter_type == "Median":
            self.modified_image = self.median_filter(self.modified_image, filter_size)
        elif filter_type == "Gaussian":
            self.modified_image = self.gaussian_filter(self.modified_image, filter_size, sigma)

        self.show_image()

    def average_filter(self, image, filter_size):
        """Apply average filter using numpy operations."""
        kernel = np.ones((filter_size, filter_size), np.float32) / (filter_size * filter_size)
        padded_image = np.pad(image,
                              ((filter_size // 2, filter_size // 2),
                               (filter_size // 2, filter_size // 2),
                               (0, 0)),
                              mode='reflect')  # Using reflect padding for better edge handling
        filtered_image = np.zeros_like(image)

        # Loop through each channel
        for k in range(image.shape[2]):
            # Loop through each pixel
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    neighborhood = padded_image[i:i + filter_size, j:j + filter_size, k]
                    filtered_image[i, j, k] = np.sum(neighborhood * kernel)

        return filtered_image.astype(np.uint8)

    def median_filter(self, image, filter_size):
        """Apply median filter using numpy operations."""
        padded_image = np.pad(image,
                              ((filter_size // 2, filter_size // 2),
                               (filter_size // 2, filter_size // 2),
                               (0, 0)),
                              mode='reflect')  # Using reflect padding for better edge handling
        filtered_image = np.zeros_like(image)

        # Loop through each channel
        for k in range(image.shape[2]):
            # Loop through each pixel
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    neighborhood = padded_image[i:i + filter_size, j:j + filter_size, k]
                    filtered_image[i, j, k] = np.median(neighborhood)

        return filtered_image.astype(np.uint8)

    def gaussian_filter(self, image, filter_size, sigma=1.0):
        """Apply Gaussian filter using numpy operations."""
        # Ensure filter size is odd
        if filter_size % 2 == 0:
            filter_size += 1

        # Create 1D Gaussian kernel
        ax = np.linspace(-(filter_size - 1) / 2., (filter_size - 1) / 2., filter_size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel = kernel / np.sum(kernel)  # Normalize

        padded_image = np.pad(image,
                              ((filter_size // 2, filter_size // 2),
                               (filter_size // 2, filter_size // 2),
                               (0, 0)),
                              mode='reflect')  # Using reflect padding for better edge handling
        filtered_image = np.zeros_like(image)

        # Loop through each channel
        for k in range(image.shape[2]):
            # Loop through each pixel
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    neighborhood = padded_image[i:i + filter_size, j:j + filter_size, k]
                    filtered_image[i, j, k] = np.sum(neighborhood * kernel)

        return filtered_image.astype(np.uint8)


# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Image Filtering Application")
#         self.setCentralWidget(FilterNoiseWidget())
#         self.setMinimumSize(800, 600)
#
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec())