import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QWidget, QFileDialog, \
    QComboBox, QSlider, QHBoxLayout, QGroupBox
from PySide6.QtGui import QImage, QPixmap, Qt
from PySide6.QtCore import QSize
import cv2


class DetectEdgesWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.modified_image = None
        self.original_pixmap = None

        self.setWindowTitle("Edge Detection")
        self.resize(800, 600)

        # Main layout
        main_layout = QVBoxLayout()

        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        main_layout.addWidget(self.image_label)

        # Controls
        controls_layout = QHBoxLayout()

        # Left controls - Image loading
        left_controls = QGroupBox("Image")
        left_layout = QVBoxLayout()
        self.load_image_button = QPushButton("Load Image")
        self.load_image_button.clicked.connect(self.load_image)
        left_layout.addWidget(self.load_image_button)
        left_controls.setLayout(left_layout)
        controls_layout.addWidget(left_controls)

        # Middle controls - Detection method
        middle_controls = QGroupBox("Edge Detection Method")
        middle_layout = QVBoxLayout()
        self.edge_detection_menu = QComboBox()
        self.edge_detection_menu.addItems(["Sobel", "Roberts", "Prewitt", "Canny"])
        middle_layout.addWidget(self.edge_detection_menu)
        middle_controls.setLayout(middle_layout)
        controls_layout.addWidget(middle_controls)

        # Right controls - Parameters
        right_controls = QGroupBox("Parameters")
        right_layout = QVBoxLayout()

        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Threshold:")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(50)
        self.threshold_slider.setValue(20)
        self.threshold_value_label = QLabel("20")
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)

        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_value_label)
        right_layout.addLayout(threshold_layout)

        kernel_layout = QHBoxLayout()
        kernel_label = QLabel("Kernel Size:")
        self.kernel_slider = QSlider(Qt.Horizontal)
        self.kernel_slider.setMinimum(3)
        self.kernel_slider.setMaximum(7)
        self.kernel_slider.setSingleStep(2)
        self.kernel_slider.setValue(3)
        self.kernel_value_label = QLabel("3")
        self.kernel_slider.valueChanged.connect(self.update_kernel_label)

        kernel_layout.addWidget(kernel_label)
        kernel_layout.addWidget(self.kernel_slider)
        kernel_layout.addWidget(self.kernel_value_label)
        right_layout.addLayout(kernel_layout)

        self.detect_edges_button = QPushButton("Detect Edges")
        self.detect_edges_button.clicked.connect(self.detect_edges)
        right_layout.addWidget(self.detect_edges_button)

        right_controls.setLayout(right_layout)
        controls_layout.addWidget(right_controls)

        main_layout.addLayout(controls_layout)
        self.setLayout(main_layout)

    def update_threshold_label(self, value):
        self.threshold_value_label.setText(str(value))

    def update_kernel_label(self, value):
        # Ensure kernel size is odd
        if value % 2 == 0:
            value = value + 1
            self.kernel_slider.setValue(value)
        self.kernel_value_label.setText(str(value))

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.image = cv2.imread(file_name)
            self.modified_image = self.image.copy()
            self.show_image(self.modified_image)

    def show_image(self, img):
        if img is None:
            return

        if len(img.shape) == 2:
            # Grayscale image
            h, w = img.shape
            qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        else:
            # Color image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg)

        # Scale pixmap to fit in label while maintaining aspect ratio
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
        self.original_pixmap = pixmap

    def detect_edges(self):
        if self.image is None:
            return

        # Get parameters
        edge_detection = self.edge_detection_menu.currentText()
        threshold = self.threshold_slider.value()
        kernel_size = self.kernel_slider.value()

        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Convert to grayscale if needed
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image.copy()

        # Apply edge detection
        if edge_detection == "Sobel":
            self.modified_image = self.sobel_edge_detection(gray, threshold, kernel_size)
        elif edge_detection == "Roberts":
            self.modified_image = self.roberts_edge_detection(gray, threshold)
        elif edge_detection == "Prewitt":
            self.modified_image = self.prewitt_edge_detection(gray, threshold, kernel_size)
        elif edge_detection == "Canny":
            low_threshold = threshold
            high_threshold = threshold * 3
            self.modified_image = cv2.Canny(self.image, low_threshold, high_threshold)

        self.show_image(self.modified_image)

    def sobel_edge_detection(self, image, threshold=20, kernel_size=3):
        """Apply Sobel edge detection from scratch with variable kernel size"""
        # Get image dimensions
        rows, cols = image.shape

        # Create Sobel kernels based on kernel size
        if kernel_size == 3:
            Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        else:
            # Generate larger Sobel-like kernels
            # This is a simple approximation - more sophisticated kernels could be used
            mid = kernel_size // 2
            Kx = np.zeros((kernel_size, kernel_size))
            Ky = np.zeros((kernel_size, kernel_size))

            for i in range(kernel_size):
                for j in range(kernel_size):
                    x_dist = j - mid
                    y_dist = i - mid
                    if x_dist != 0:
                        Kx[i, j] = x_dist / abs(x_dist) * (mid - abs(y_dist)) if abs(y_dist) <= mid else 0
                    if y_dist != 0:
                        Ky[i, j] = y_dist / abs(y_dist) * (mid - abs(x_dist)) if abs(x_dist) <= mid else 0

        # Initialize gradient images
        Ix = np.zeros_like(image, dtype=np.float64)
        Iy = np.zeros_like(image, dtype=np.float64)

        # Apply convolution
        pad = kernel_size // 2
        padded_image = np.pad(image, pad, mode='constant')

        for i in range(rows):
            for j in range(cols):
                region = padded_image[i:i + kernel_size, j:j + kernel_size]
                Ix[i, j] = np.sum(region * Kx)
                Iy[i, j] = np.sum(region * Ky)

        # Calculate gradient magnitude
        G = np.sqrt(Ix ** 2 + Iy ** 2)
        G = np.clip(G, 0, 255).astype(np.uint8)

        # Apply threshold
        _, G = cv2.threshold(G, threshold, 255, cv2.THRESH_BINARY)

        return G

    def roberts_edge_detection(self, image, threshold=20):
        """Apply Roberts Cross edge detection from scratch"""
        # Roberts operators
        Kx = np.array([[1, 0], [0, -1]])
        Ky = np.array([[0, 1], [-1, 0]])

        # Get image dimensions
        rows, cols = image.shape

        # Initialize output
        G = np.zeros_like(image, dtype=np.float64)

        # Apply operators
        for i in range(rows - 1):
            for j in range(cols - 1):
                # Extract 2x2 region
                region = image[i:i + 2, j:j + 2]
                # Apply Roberts operators
                gx = np.sum(region * Kx)
                gy = np.sum(region * Ky)
                # Calculate magnitude
                G[i, j] = np.sqrt(gx ** 2 + gy ** 2)

        # Normalize and convert to uint8
        G = np.clip(G, 0, 255).astype(np.uint8)

        # Apply threshold
        _, G = cv2.threshold(G, threshold, 255, cv2.THRESH_BINARY)

        return G

    def prewitt_edge_detection(self, image, threshold=20, kernel_size=3):
        """Apply Prewitt edge detection from scratch with variable kernel size"""
        # Get image dimensions
        rows, cols = image.shape

        # Create Prewitt kernels based on kernel size
        if kernel_size == 3:
            Kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            Ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        else:
            # Generate larger Prewitt-like kernels
            Kx = np.zeros((kernel_size, kernel_size))
            Ky = np.zeros((kernel_size, kernel_size))
            mid = kernel_size // 2

            for i in range(kernel_size):
                for j in range(kernel_size):
                    if j < mid:
                        Kx[i, j] = -1
                    elif j > mid:
                        Kx[i, j] = 1

                    if i < mid:
                        Ky[i, j] = -1
                    elif i > mid:
                        Ky[i, j] = 1

        # Initialize gradient images
        Ix = np.zeros_like(image, dtype=np.float64)
        Iy = np.zeros_like(image, dtype=np.float64)

        # Apply convolution
        pad = kernel_size // 2
        padded_image = np.pad(image, pad, mode='constant')

        for i in range(rows):
            for j in range(cols):
                region = padded_image[i:i + kernel_size, j:j + kernel_size]
                Ix[i, j] = np.sum(region * Kx)
                Iy[i, j] = np.sum(region * Ky)

        # Calculate gradient magnitude
        G = np.sqrt(Ix ** 2 + Iy ** 2)
        G = np.clip(G, 0, 255).astype(np.uint8)

        # Apply threshold
        _, G = cv2.threshold(G, threshold, 255, cv2.THRESH_BINARY)

        return G


# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Edge Detection")
#         self.setCentralWidget(DetectEdgesWidget())
#         self.resize(800, 600)
#
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec())