import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QFileDialog, QFrame
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
import cv2
class EqualizingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.modified_image = None
        self.setWindowTitle("Equalize")
        self.layout = QVBoxLayout()
        
        # handling images and their layout
        self.images_frame= QFrame()
        self.images_layout = QHBoxLayout()
        self.original_frame = QFrame()
        
        """"needs a selected image set with specific dimensions 34an el stretching/trimming,
        m4 bt3ml ma4akel bas btdrab warnings fl terminal 3l fady ya3ni law el dimensions akbar mn el max size.
        """
        self.original_label = QLabel("Original")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMaximumSize(800, 100)
        self.original_layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setMaximumSize(600, 500)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.original_layout.addWidget(self.original_label)
        self.original_layout.addWidget(self.image_label)
        self.original_frame.setLayout(self.original_layout)
        self.images_layout.addWidget(self.original_frame)
        self.modified_frame = QFrame()
        self.modified_label = QLabel("Equalized")
        self.modified_label.setAlignment(Qt.AlignCenter)
        self.modified_label.setMaximumSize(800, 100)
        self.modified_layout = QVBoxLayout()
        self.modified_image_label = QLabel()
        self.modified_image_label.setMaximumSize(600, 500)
        self.modified_image_label.setAlignment(Qt.AlignCenter)
        self.modified_layout.addWidget(self.modified_label)
        self.modified_layout.addWidget(self.modified_image_label)
        self.modified_frame.setLayout(self.modified_layout)
        self.images_layout.addWidget(self.modified_frame)
        self.images_frame.setLayout(self.images_layout)
        self.layout.addWidget(self.images_frame)
        
        # add button to load image
        self.load_image_button = QPushButton("Load Image")
        self.load_image_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_image_button)
        
        # add button to equalize histogram
        self.equalize_button = QPushButton("Equalize")
        self.equalize_button.clicked.connect(self.equalize)
        self.layout.addWidget(self.equalize_button)
        
        self.setLayout(self.layout)
        
    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.image = cv2.imread(file_name)
            self.modified_image = self.image
            self.modified_image_label.clear()
            self.show_image("original")
    
    def show_image(self, type):
        if len(self.modified_image.shape) == 3 and self.modified_image.shape[2] == 3:
            self.modified_image = cv2.cvtColor(self.modified_image, cv2.COLOR_BGR2RGB)
        h, w, ch = self.modified_image.shape
        bytes_per_line = ch * w
        qimage = QImage(self.modified_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        if type == "original":
            self.image_label.setPixmap(pixmap)
        else:
            self.modified_image_label.setPixmap(pixmap)
    
    def equalize(self):
        if self.image is None:
            return

        # RGB to greyscale
        gray_image = np.dot(self.image[...,:3], [0.114, 0.587, 0.299]).astype(np.uint8)

        # equalization (not sure yet if the result should be greyscale or RGB)
        hist, bins = np.histogram(gray_image.flatten(), bins=256, range=[0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * 255 / cdf[-1]
        equalized_array = cdf_normalized[gray_image].astype(np.uint8)

        # greyscale to RGB
        self.modified_image = np.stack((equalized_array,) * 3, axis=-1)  

        self.show_image("modified")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Equalize")
        self.setCentralWidget(EqualizingWidget())

app = QApplication([])
window = MainWindow()
window.show()
sys.exit(app.exec())