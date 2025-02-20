# create a pyside6 application with widget that detect edges of an image
# implement all the masks from scratch
# types of masks: sobel, roberts, prewitt and canny edge detection

import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QWidget, QFileDialog, \
    QComboBox, QSlider
from PySide6.QtGui import QImage, QPixmap, Qt
import cv2

class DetectEdgesWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.modified_image = self.image
        self.setWindowTitle("Detect Edges")
        self.layout = QVBoxLayout()
        self.image_label = QLabel()
        # temporary fix for size issues
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMaximumSize(600, 400)
        
        self.layout.addWidget(self.image_label)
        # add button to load image
        self.load_image_button = QPushButton("Load Image")
        self.load_image_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_image_button)
        # add dropdown menu to select the type of edge detection (sobel, roberts, prewitt, canny)
        self.edge_detection_menu = QComboBox()
        self.edge_detection_menu.addItems(["Sobel", "Roberts", "Prewitt", "Canny"])
        self.layout.addWidget(self.edge_detection_menu)
        # add slider to adjust the threshold
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(15)
        self.threshold_slider.setValue(10)
        self.layout.addWidget(self.threshold_slider)
        # add button to detect edges
        self.detect_edges_button = QPushButton("Detect Edges")
        self.detect_edges_button.clicked.connect(self.detect_edges)
        self.layout.addWidget(self.detect_edges_button)
        self.setLayout(self.layout)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.image = cv2.imread(file_name)
            self.modified_image = self.image
            self.show_image()

    def show_image(self):
        if self.modified_image is None:
            return
        if len(self.modified_image.shape) == 2:
            # Grayscale image
            h, w = self.modified_image.shape
            qimage = QImage(self.modified_image.data, w, h, w, QImage.Format_Grayscale8)
        elif len(self.modified_image.shape) == 3 and self.modified_image.shape[2] == 3:
            # Color image
            self.modified_image = cv2.cvtColor(self.modified_image, cv2.COLOR_BGR2RGB)
            h, w, ch = self.modified_image.shape
            bytes_per_line = ch * w
            qimage = QImage(self.modified_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            return
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap)
        
        # temporary fix for size issues
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)  

    def detect_edges(self):
        if self.image is None:
            return
        edge_detection = self.edge_detection_menu.currentText()
        threshold = self.threshold_slider.value()
        if edge_detection == "Sobel":
            self.modified_image = self.sobel_edge_detection(self.image, threshold)
        elif edge_detection == "Roberts":
            self.modified_image = self.roberts_edge_detection(self.image)
        elif edge_detection == "Prewitt":
            self.modified_image = self.prewitt_edge_detection(self.image)
        elif edge_detection == "Canny":
            self.modified_image = self.canny_edge_detection(self.image)
        self.show_image()

    def sobel_edge_detection(self, image, threshold=5):
        # Convert to grayscale if the image is in color
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        # Get image dimensions
        rows, cols = image.shape

        # Initialize gradient images
        Ix = np.zeros_like(image)
        Iy = np.zeros_like(image)

        # Apply Sobel kernels
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                region = image[i - 1:i + 2, j - 1:j + 2]
                Ix[i, j] = np.sum(region * Kx)
                Iy[i, j] = np.sum(region * Ky)

        # Calculate gradient magnitude
        G = np.sqrt(Ix ** 2 + Iy ** 2)
        G = np.clip(G, 0, 255).astype(np.uint8)

        # Apply threshold
        _, G = cv2.threshold(G, threshold, 255, cv2.THRESH_BINARY)

        return G

    def roberts_edge_detection(self, image):
        Kx = np.array([[1, 0], [0, -1]])
        Ky = np.array([[0, 1], [-1, 0]])
        Ix = cv2.filter2D(image, -1, Kx)
        Iy = cv2.filter2D(image, -1, Ky)
        G = np.sqrt(Ix ** 2 + Iy ** 2)
        G = np.clip(G, 0, 255).astype(np.uint8)
        return G

    def prewitt_edge_detection(self, image):
        Kx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        Ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        Ix = cv2.filter2D(image, -1, Kx)
        Iy = cv2.filter2D(image, -1, Ky)
        G = np.sqrt(Ix ** 2 + Iy ** 2)
        G = np.clip(G, 0, 255).astype(np.uint8)
        return G

    def canny_edge_detection(self, image, low_threshold=50, high_threshold=150):
        edges = cv2.Canny(image, low_threshold, high_threshold)
        return edges


# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Detect Edges")
#         self.setCentralWidget(DetectEdgesWidget())

# app = QApplication([])
# window = MainWindow()
# window.show()
# sys.exit(app.exec())


