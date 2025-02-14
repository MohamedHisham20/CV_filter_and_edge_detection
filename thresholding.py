import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QFileDialog, QFrame, QSlider
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
import cv2

class ThresholdingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.modified_image = None
        self.setWindowTitle("Thresholding")
        self.layout = QVBoxLayout()
        
        # handling images and their layout
        self.images_frame= QFrame()
        self.images_layout = QHBoxLayout()
        self.original_frame = QFrame()
       
        self.original_label = QLabel("Input")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMaximumSize(800, 100)
        self.original_layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setMaximumSize(800, 500)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.original_layout.addWidget(self.original_label)
        self.original_layout.addWidget(self.image_label)
        self.original_frame.setLayout(self.original_layout)
        self.images_layout.addWidget(self.original_frame)
        self.modified_frame = QFrame()
        self.modified_label = QLabel("Output")
        self.modified_label.setAlignment(Qt.AlignCenter)
        self.modified_label.setMaximumSize(600, 100)
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
        
        # add slider to adjust the threshold value
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(255)
        self.threshold_slider.setValue(128)
        self.layout.addWidget(self.threshold_slider)
        
        # add button for local thresholding
        self.local_button = QPushButton("Local")
        self.local_button.clicked.connect(self.local_thresholding)
        self.layout.addWidget(self.local_button)
        
        # add button for global thresholding
        self.global_button = QPushButton("Global")
        self.global_button.clicked.connect(self.global_thresholding)
        self.layout.addWidget(self.global_button)
        
        self.setLayout(self.layout)
        
    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)  
            self.modified_image = self.image.copy() 
            self.show_image(self.image_label, self.image)  
    
    def show_image(self, label, image):
        height, width = image.shape
        bytes_per_line = width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        label.setPixmap(pixmap)
        label.setScaledContents(True)
            
    def local_thresholding(self):
        pass
        
    def global_thresholding(self):
        pass
            
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Equalize")
        self.setCentralWidget(ThresholdingWidget())

app = QApplication([])
window = MainWindow()
window.show()
sys.exit(app.exec())