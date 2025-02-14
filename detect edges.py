# create a pyside6 application with widget that detect edges of an image
# implement all the masks from scratch
# types of masks: sobel, roberts, prewitt and canny edge detection

import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QWidget, QFileDialog, \
    QComboBox, QSlider
from PySide6.QtGui import QImage, QPixmap
import cv2

class DetectEdgesWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.modified_image = self.image
        self.setWindowTitle("Detect Edges")
        self.layout = QVBoxLayout()
        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)
        # add button to load image
        self.load_image_button = QPushButton("Load Image")
        self.load_image_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_image_button)
        # add dropdown menu to select the type of edge detection (sobel, roberts, prewitt, canny)
        self.edge_detection_menu = QComboBox()
        self.edge_detection_menu.addItems(["Sobel", "Roberts", "Prewitt", "Canny"])
        self.layout.addWidget(self.edge_detection_menu)
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
        if len(self.modified_image.shape) == 3 and self.modified_image.shape[2] == 3:
            self.modified_image = cv2.cvtColor(self.modified_image, cv2.COLOR_BGR2RGB)
        h, w, ch = self.modified_image.shape
        bytes_per_line = ch * w
        qimage = QImage(self.modified_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap)

    def detect_edges(self):
        if self.image is None:
            return
        edge_detection = self.edge_detection_menu.currentText()
        if edge_detection == "Sobel":
            self.modified_image = self.sobel_edge_detection()
        elif edge_detection == "Roberts":
            self.modified_image = self.roberts_edge_detection()
        elif edge_detection == "Prewitt":
            self.modified_image = self.prewitt_edge_detection()
        elif edge_detection == "Canny":
            self.modified_image = self.canny_edge_detection()
        self.show_image()

    def sobel_edge_detection(image):
        Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        Ix = cv2.filter2D(image, -1, Kx)
        Iy = cv2.filter2D(image, -1, Ky)
        G = np.sqrt(Ix ** 2 + Iy ** 2)
        G = np.clip(G, 0, 255).astype(np.uint8)
        return G

    def roberts_edge_detection(image):
        Kx = np.array([[1, 0], [0, -1]])
        Ky = np.array([[0, 1], [-1, 0]])
        Ix = cv2.filter2D(image, -1, Kx)
        Iy = cv2.filter2D(image, -1, Ky)
        G = np.sqrt(Ix ** 2 + Iy ** 2)
        G = np.clip(G, 0, 255).astype(np.uint8)
        return G

    def prewitt_edge_detection(image):
        Kx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        Ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        Ix = cv2.filter2D(image, -1, Kx)
        Iy = cv2.filter2D(image, -1, Ky)
        G = np.sqrt(Ix ** 2 + Iy ** 2)
        G = np.clip(G, 0, 255).astype(np.uint8)
        return G

    def canny_edge_detection(image, low_threshold=50, high_threshold=150):
        # Step 1: Noise reduction using Gaussian filter
        image = cv2.GaussianBlur(image, (5, 5), 1.4)

        # Step 2: Gradient calculation using Sobel operator
        Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        Ix = cv2.filter2D(image, -1, Kx)
        Iy = cv2.filter2D(image, -1, Ky)
        G = np.sqrt(Ix ** 2 + Iy ** 2)
        theta = np.arctan2(Iy, Ix)

        # Step 3: Non-maximum suppression
        M, N = G.shape
        Z = np.zeros((M, N), dtype=np.int32)
        angle = theta * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                try:
                    q = 255
                    r = 255

                    # angle 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = G[i, j + 1]
                        r = G[i, j - 1]
                    # angle 45
                    elif 22.5 <= angle[i, j] < 67.5:
                        q = G[i + 1, j - 1]
                        r = G[i - 1, j + 1]
                    # angle 90
                    elif 67.5 <= angle[i, j] < 112.5:
                        q = G[i + 1, j]
                        r = G[i - 1, j]
                    # angle 135
                    elif 112.5 <= angle[i, j] < 157.5:
                        q = G[i - 1, j - 1]
                        r = G[i + 1, j + 1]

                    if (G[i, j] >= q) and (G[i, j] >= r):
                        Z[i, j] = G[i, j]
                    else:
                        Z[i, j] = 0

                except IndexError as e:
                    pass

        # Step 4: Edge tracking by hysteresis
        weak = 50
        strong = 255
        result = np.zeros_like(G)
        strong_i, strong_j = np.where(Z >= high_threshold)
        zeros_i, zeros_j = np.where(Z < low_threshold)
        weak_i, weak_j = np.where((Z <= high_threshold) & (Z >= low_threshold))

        result[strong_i, strong_j] = strong
        result[weak_i, weak_j] = weak

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if (result[i, j] == weak):
                    if ((result[i + 1, j - 1] == strong) or (result[i + 1, j] == strong) or (
                            result[i + 1, j + 1] == strong)
                            or (result[i, j - 1] == strong) or (result[i, j + 1] == strong)
                            or (result[i - 1, j - 1] == strong) or (result[i - 1, j] == strong) or (
                                    result[i - 1, j + 1] == strong)):
                        result[i, j] = strong
                    else:
                        result[i, j] = 0

        return result

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detect Edges")
        self.setCentralWidget(DetectEdgesWidget())

app = QApplication([])
window = MainWindow()
window.show()
sys.exit(app.exec())


