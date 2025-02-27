import numpy as np
from PySide6.QtWidgets import QLabel, QVBoxLayout, QPushButton, QWidget, QFileDialog, QComboBox, QSlider, \
    QHBoxLayout, QCheckBox
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
import cv2



class AddNoiseWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.original_image = None  # Keep a copy of the original image
        self.modified_image = None
        self.setWindowTitle("Add Noise")

        # Main layout
        self.layout = QVBoxLayout()

        # Image display
        self.image_label = QLabel()
        self.image_label.setMinimumSize(600, 400)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        # Load image button
        button_layout = QHBoxLayout()
        self.load_image_button = QPushButton("Load Image")
        self.load_image_button.clicked.connect(self.load_image)
        button_layout.addWidget(self.load_image_button)

        # Reset image button
        self.reset_button = QPushButton("Reset Image")
        self.reset_button.clicked.connect(self.reset_image)
        button_layout.addWidget(self.reset_button)
        self.layout.addLayout(button_layout)

        # Noise type selector
        noise_type_layout = QHBoxLayout()
        noise_type_layout.addWidget(QLabel("Noise Type:"))
        self.noise_type_menu = QComboBox()
        self.noise_type_menu.addItems([
            "Gaussian (From Scratch)",
            "Salt and Pepper (From Scratch)",
            "Uniform (From Scratch)",
            "Gaussian (OpenCV)",
            "Salt and Pepper (OpenCV)",
            "Uniform (OpenCV)"
        ])
        noise_type_layout.addWidget(self.noise_type_menu)
        self.layout.addLayout(noise_type_layout)

        # Noise parameter sliders
        # Noise level/intensity
        noise_level_layout = QHBoxLayout()
        noise_level_layout.addWidget(QLabel("Noise Intensity:"))
        self.noise_level_slider = QSlider(Qt.Horizontal)
        self.noise_level_slider.setMinimum(0)
        self.noise_level_slider.setMaximum(100)
        self.noise_level_slider.setValue(25)  # Default to lower value
        self.noise_level_label = QLabel("25%")
        self.noise_level_slider.valueChanged.connect(self.update_noise_level_label)
        noise_level_layout.addWidget(self.noise_level_slider)
        noise_level_layout.addWidget(self.noise_level_label)
        self.layout.addLayout(noise_level_layout)

        # For Gaussian noise - standard deviation
        std_layout = QHBoxLayout()
        std_layout.addWidget(QLabel("Standard Deviation (Gaussian):"))
        self.std_slider = QSlider(Qt.Horizontal)
        self.std_slider.setMinimum(1)
        self.std_slider.setMaximum(100)
        self.std_slider.setValue(25)
        self.std_label = QLabel("25")
        self.std_slider.valueChanged.connect(self.update_std_label)
        std_layout.addWidget(self.std_slider)
        std_layout.addWidget(self.std_label)
        self.layout.addLayout(std_layout)

        # Salt and Pepper probability
        sp_layout = QHBoxLayout()
        sp_layout.addWidget(QLabel("Salt and Pepper Ratio:"))
        self.sp_slider = QSlider(Qt.Horizontal)
        self.sp_slider.setMinimum(1)
        self.sp_slider.setMaximum(50)  # Max 50% for S&P noise
        self.sp_slider.setValue(5)
        self.sp_label = QLabel("5%")
        self.sp_slider.valueChanged.connect(self.update_sp_label)
        sp_layout.addWidget(self.sp_slider)
        sp_layout.addWidget(self.sp_label)
        self.layout.addLayout(sp_layout)

        # Checkbox to preserve the original noise image
        preserve_layout = QHBoxLayout()
        self.preserve_checkbox = QCheckBox("Save original noisy image for comparison")
        self.preserve_checkbox.setChecked(True)
        preserve_layout.addWidget(self.preserve_checkbox)
        self.layout.addLayout(preserve_layout)

        # Add noise button
        action_layout = QHBoxLayout()
        self.add_noise_button = QPushButton("Add Noise")
        self.add_noise_button.clicked.connect(self.add_noise)
        action_layout.addWidget(self.add_noise_button)

        # Save image button
        self.save_image_button = QPushButton("Save Image")
        self.save_image_button.clicked.connect(self.save_image)
        action_layout.addWidget(self.save_image_button)
        self.layout.addLayout(action_layout)

        self.setLayout(self.layout)

        # Initialize state
        self.update_noise_level_label(self.noise_level_slider.value())
        self.update_std_label(self.std_slider.value())
        self.update_sp_label(self.sp_slider.value())

        # Store the last used noise parameters for comparison
        self.last_noise_type = None
        self.last_noise_params = {}

        # For storing the noisy image (to compare with filtered results)
        self.noisy_image = None

    def update_noise_level_label(self, value):
        self.noise_level_label.setText(f"{value}%")

    def update_std_label(self, value):
        self.std_label.setText(str(value))

    def update_sp_label(self, value):
        self.sp_label.setText(f"{value}%")

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            # Load and convert to RGB immediately
            self.original_image = cv2.imread(file_name)
            if self.original_image is None:
                return

            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.image = self.original_image.copy()
            self.modified_image = self.image.copy()
            self.show_image()

    def reset_image(self):
        if self.original_image is not None:
            self.image = self.original_image.copy()
            self.modified_image = self.image.copy()
            self.noisy_image = None  # Clear the stored noisy image
            self.show_image()

    def show_image(self):
        if self.modified_image is None:
            return

        h, w, ch = self.modified_image.shape
        bytes_per_line = ch * w
        qimage = QImage(self.modified_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        # Scale pixmap to fit in label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(self.image_label.size(),
                                      Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation)

        self.image_label.setPixmap(scaled_pixmap)

    def save_image(self):
        if self.modified_image is None:
            return

        file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            # Convert the image back to BGR format before saving
            image_to_save = cv2.cvtColor(self.modified_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_name, image_to_save)

            # If we have a noisy image and want to save it too
            if self.noisy_image is not None and self.preserve_checkbox.isChecked():
                # Add "_noisy" suffix to the filename
                noisy_file_name = file_name.split('.')
                noisy_file_name = noisy_file_name[0] + "_noisy." + noisy_file_name[1]
                noisy_image_to_save = cv2.cvtColor(self.noisy_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(noisy_file_name, noisy_image_to_save)

    def add_noise(self):
        if self.image is None:
            return

        noise_type = self.noise_type_menu.currentText()
        noise_level = self.noise_level_slider.value() / 100.0
        std_dev = self.std_slider.value()
        sp_ratio = self.sp_slider.value() / 100.0

        # Store parameters for comparison
        self.last_noise_type = noise_type
        self.last_noise_params = {
            'noise_level': noise_level,
            'std_dev': std_dev,
            'sp_ratio': sp_ratio
        }

        # Apply the selected noise
        if "Gaussian (From Scratch)" in noise_type:
            self.modified_image = self.add_gaussian_noise(self.image.copy(), std_dev, noise_level)
        elif "Salt and Pepper (From Scratch)" in noise_type:
            self.modified_image = self.add_salt_and_pepper_noise(self.image.copy(), sp_ratio)
        elif "Uniform (From Scratch)" in noise_type:
            self.modified_image = self.add_uniform_noise(self.image.copy(), noise_level)
        elif "Gaussian (OpenCV)" in noise_type:
            self.modified_image = self.add_gaussian_noise_openCV(self.image.copy(), 0, std_dev)
        elif "Salt and Pepper (OpenCV)" in noise_type:
            self.modified_image = self.add_salt_and_pepper_noise_openCV(self.image.copy(), sp_ratio)
        elif "Uniform (OpenCV)" in noise_type:
            self.modified_image = self.add_uniform_noise_openCV(self.image.copy(), noise_level)

        # Store the noisy image for comparison with filtered results
        if self.preserve_checkbox.isChecked():
            self.noisy_image = self.modified_image.copy()

        self.show_image()

    def add_salt_and_pepper_noise(self, image, ratio):
        """Add salt and pepper noise from scratch"""
        noisy_image = np.copy(image)

        # Create masks for salt and pepper
        salt_mask = np.random.random(image.shape[:2]) < ratio / 2
        pepper_mask = np.random.random(image.shape[:2]) < ratio / 2

        # Expand masks to match image dimensions
        salt_mask = np.repeat(salt_mask[:, :, np.newaxis], 3, axis=2)
        pepper_mask = np.repeat(pepper_mask[:, :, np.newaxis], 3, axis=2)

        # Add salt (white) and pepper (black) noise
        noisy_image[salt_mask] = 255
        noisy_image[pepper_mask & ~salt_mask] = 0  # Only apply pepper where salt isn't

        return noisy_image.astype(np.uint8)

    def add_salt_and_pepper_noise_openCV(self, image, ratio):
        """Add salt and pepper noise using OpenCV approach"""
        noisy_image = np.copy(image)

        # Calculate number of salt and pepper points
        h, w, c = image.shape
        total_pixels = h * w
        num_salt = int(total_pixels * ratio / 2)
        num_pepper = int(total_pixels * ratio / 2)

        # Add salt noise
        for i in range(num_salt):
            y = np.random.randint(0, h)
            x = np.random.randint(0, w)
            noisy_image[y, x, :] = 255

        # Add pepper noise
        for i in range(num_pepper):
            y = np.random.randint(0, h)
            x = np.random.randint(0, w)
            noisy_image[y, x, :] = 0

        return noisy_image

    def add_gaussian_noise(self, image, std, intensity):
        """Add Gaussian noise from scratch"""
        # Generate Gaussian noise
        gaussian_noise = np.random.normal(0, std, image.shape).astype(np.float32)

        # Scale noise by intensity factor
        scaled_noise = gaussian_noise * intensity

        # Add noise to image
        noisy_image = image.astype(np.float32) + scaled_noise

        # Clip values to valid range
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

        return noisy_image

    def add_gaussian_noise_openCV(self, image, mean=0, std=25):
        """Add Gaussian noise using OpenCV approach"""
        # Generate Gaussian noise with the same shape as the image
        gaussian = np.random.normal(mean, std, image.shape).astype(np.float32)

        # Add the noise to the image
        noisy_image = cv2.add(image.astype(np.float32), gaussian)

        # Clip values to valid range
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

        return noisy_image

    def add_uniform_noise(self, image, intensity):
        """Add uniform noise from scratch"""
        # Generate uniform noise in the range [-intensity*255, intensity*255]
        range_val = intensity * 255
        uniform_noise = np.random.uniform(-range_val, range_val, image.shape).astype(np.float32)

        # Add noise to image
        noisy_image = image.astype(np.float32) + uniform_noise

        # Clip values to valid range
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

        return noisy_image

    def add_uniform_noise_openCV(self, image, intensity):
        """Add uniform noise using OpenCV approach"""
        # Generate uniform noise
        range_val = intensity * 255
        uniform_noise = np.random.uniform(-range_val / 2, range_val / 2, image.shape).astype(np.float32)

        # Add the noise to the image using cv2.add
        noisy_image = cv2.add(image.astype(np.float32), uniform_noise)

        # Clip values to valid range
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

        return noisy_image


# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Add Noise")
#         self.setCentralWidget(AddNoiseWidget())
#         self.setMinimumSize(800, 600)
#
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec())