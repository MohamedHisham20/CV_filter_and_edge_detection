import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QFileDialog, QFrame,
                               QComboBox, QSpinBox)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
import cv2


class HybridImagesWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.image1 = cv2.imread("images/submarine.bmp")
        self.image2 = cv2.imread("images/fish.bmp")
        self.low_passed = None
        self.high_passed = None
        self.hybrid = None

        self.setWindowTitle("Hybrid Images")
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout()

        # Top panel containing input images and controls
        top_panel = QWidget()
        top_layout = QHBoxLayout()

        # Input images panel
        input_panel = QWidget()
        input_layout = QHBoxLayout()

        # First image (low-pass) panel
        self.image1_frame = QFrame()
        image1_layout = QVBoxLayout()
        self.image1_label = QLabel("Image 1 (Low-passed)")
        self.image1_label.setAlignment(Qt.AlignCenter)
        self.image1_display = QLabel()
        self.image1_display.setMinimumSize(400, 300)
        self.image1_display.setAlignment(Qt.AlignCenter)
        self.load_image1_btn = QPushButton("Load Image 1")
        self.load_image1_btn.clicked.connect(self.load_image1)
        image1_layout.addWidget(self.image1_label)
        image1_layout.addWidget(self.image1_display)
        image1_layout.addWidget(self.load_image1_btn)
        self.image1_frame.setLayout(image1_layout)

        # Second image (high-pass) panel
        self.image2_frame = QFrame()
        image2_layout = QVBoxLayout()
        self.image2_label = QLabel("Image 2 (High-passed)")
        self.image2_label.setAlignment(Qt.AlignCenter)
        self.image2_display = QLabel()
        self.image2_display.setMinimumSize(400, 300)
        self.image2_display.setAlignment(Qt.AlignCenter)
        self.load_image2_btn = QPushButton("Load Image 2")
        self.load_image2_btn.clicked.connect(self.load_image2)
        image2_layout.addWidget(self.image2_label)
        image2_layout.addWidget(self.image2_display)
        image2_layout.addWidget(self.load_image2_btn)
        self.image2_frame.setLayout(image2_layout)

        input_layout.addWidget(self.image1_frame)
        input_layout.addWidget(self.image2_frame)
        input_panel.setLayout(input_layout)

        # Controls panel (narrow, on the right)
        controls_panel = QFrame()
        controls_panel.setMaximumWidth(200)  # Set fixed width for controls
        controls_layout = QVBoxLayout()
        controls_layout.setAlignment(Qt.AlignTop)

        # Filter type selection
        filter_type_label = QLabel("Filter Type:")
        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItems(["Gaussian", "Average"])
        controls_layout.addWidget(filter_type_label)
        controls_layout.addWidget(self.filter_type_combo)

        # Add some spacing
        controls_layout.addSpacing(10)

        # Kernel size spinbox
        kernel_label = QLabel("Kernel Size:")
        self.kernel_spinbox = QSpinBox()
        self.kernel_spinbox.setMinimum(1)
        self.kernel_spinbox.setMaximum(15)
        self.kernel_spinbox.setValue(5)
        self.kernel_spinbox.setSingleStep(2)  # Step by 2 to maintain odd numbers
        self.kernel_spinbox.valueChanged.connect(self.update_kernel_value)
        controls_layout.addWidget(kernel_label)
        controls_layout.addWidget(self.kernel_spinbox)

        # Add some spacing
        controls_layout.addSpacing(20)

        # Process button
        self.process_btn = QPushButton("Process Images")
        self.process_btn.clicked.connect(self.process_images)
        controls_layout.addWidget(self.process_btn)

        controls_panel.setLayout(controls_layout)

        # Add input panel and controls to top layout
        top_layout.addWidget(input_panel)
        top_layout.addWidget(controls_panel)
        top_panel.setLayout(top_layout)

        # Add top panel to main layout
        main_layout.addWidget(top_panel)

        # Results panel
        results_panel = QFrame()
        results_layout = QHBoxLayout()

        # Low-pass result
        low_pass_layout = QVBoxLayout()
        self.low_pass_label = QLabel("Low-pass Result")
        self.low_pass_label.setAlignment(Qt.AlignCenter)
        self.low_pass_display = QLabel()
        self.low_pass_display.setMinimumSize(300, 200)
        self.low_pass_display.setAlignment(Qt.AlignCenter)
        low_pass_layout.addWidget(self.low_pass_label)
        low_pass_layout.addWidget(self.low_pass_display)

        # High-pass result
        high_pass_layout = QVBoxLayout()
        self.high_pass_label = QLabel("High-pass Result")
        self.high_pass_label.setAlignment(Qt.AlignCenter)
        self.high_pass_display = QLabel()
        self.high_pass_display.setMinimumSize(300, 200)
        self.high_pass_display.setAlignment(Qt.AlignCenter)
        high_pass_layout.addWidget(self.high_pass_label)
        high_pass_layout.addWidget(self.high_pass_display)

        # Hybrid result
        hybrid_layout = QVBoxLayout()
        self.hybrid_label = QLabel("Hybrid Result")
        self.hybrid_label.setAlignment(Qt.AlignCenter)
        self.hybrid_display = QLabel()
        self.hybrid_display.setMinimumSize(300, 200)
        self.hybrid_display.setAlignment(Qt.AlignCenter)
        hybrid_layout.addWidget(self.hybrid_label)
        hybrid_layout.addWidget(self.hybrid_display)

        results_layout.addLayout(low_pass_layout)
        results_layout.addLayout(high_pass_layout)
        results_layout.addLayout(hybrid_layout)
        results_panel.setLayout(results_layout)
        main_layout.addWidget(results_panel)

        self.setLayout(main_layout)

        self.display_image(self.image1, self.image1_display)
        self.display_image(self.image2, self.image2_display)
        self.process_images()

    def update_kernel_value(self, value):
        # Ensure kernel size is odd
        if value % 2 == 0:
            value += 1
            self.kernel_spinbox.setValue(value)

    def load_image1(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image 1", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.image1 = cv2.imread(file_name)
            if self.image1 is not None:
                self.display_image(self.image1, self.image1_display)

    def load_image2(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image 2", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.image2 = cv2.imread(file_name)
            if self.image2 is not None:
                self.display_image(self.image2, self.image2_display)

    def display_image(self, image, label):
        if image is None:
            return

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # Scale pixmap to fit in label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)

    def process_images(self):
        if self.image1 is None or self.image2 is None:
            return

        # Get parameters
        kernel_size = self.kernel_spinbox.value()
        filter_type = self.filter_type_combo.currentText()

        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Apply low-pass filter to image1
        if filter_type == "Gaussian":
            self.low_passed = cv2.GaussianBlur(self.image1, (kernel_size, kernel_size), 0)
        else:  # Average filter
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            self.low_passed = cv2.filter2D(self.image1, -1, kernel)

        # Apply high-pass filter to image2 (original - low pass)
        if filter_type == "Gaussian":
            low_freq = cv2.GaussianBlur(self.image2, (kernel_size, kernel_size), 0)
        else:
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            low_freq = cv2.filter2D(self.image2, -1, kernel)

        self.high_passed = cv2.subtract(self.image2, low_freq)

        # Create hybrid image
        self.hybrid = cv2.add(self.low_passed, self.high_passed)

        # Display results
        self.display_image(self.low_passed, self.low_pass_display)
        self.display_image(self.high_passed, self.high_pass_display)
        self.display_image(self.hybrid, self.hybrid_display)


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    widget = HybridImagesWidget()
    widget.show()
    sys.exit(app.exec_())