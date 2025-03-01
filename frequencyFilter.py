from PySide6.QtCore import Qt, QPoint, Signal, QRect
from PySide6.QtGui import QImage, QPixmap, QPen, QPainter, QColor
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QHBoxLayout,
                               QPushButton, QFrame, QFileDialog, QButtonGroup, QDialog, QRadioButton)
import numpy as np
import cv2
import sys
import math


class FrequencyFilterWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.modified_image = None
        self.setWindowTitle("Frequency Domain Filters")
        self.layout = QVBoxLayout()

        # handling images and their layout
        self.images_frame = QFrame()
        self.images_layout = QHBoxLayout()

        # Original image panel
        self.original_frame = QFrame()
        self.original_frame.setMaximumSize(600, 700)
        self.original_label = QLabel("Original")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.original_layout.addWidget(self.original_label)
        self.original_layout.addWidget(self.image_label)
        self.original_frame.setLayout(self.original_layout)
        self.images_layout.addWidget(self.original_frame)

        # Modified image panel
        self.modified_frame = QFrame()
        self.modified_frame.setMaximumSize(600, 700)
        self.modified_label = QLabel("Filtered")
        self.modified_label.setAlignment(Qt.AlignCenter)
        self.modified_layout = QVBoxLayout()
        self.modified_image_label = QLabel()
        self.modified_image_label.setMinimumSize(400, 300)
        self.modified_image_label.setAlignment(Qt.AlignCenter)
        self.modified_layout.addWidget(self.modified_label)
        self.modified_layout.addWidget(self.modified_image_label)
        self.modified_frame.setLayout(self.modified_layout)
        self.images_layout.addWidget(self.modified_frame)

        self.images_frame.setLayout(self.images_layout)
        self.layout.addWidget(self.images_frame)

        # Control panel
        self.control_frame = QFrame()
        self.control_layout = QHBoxLayout()

        # add button to load image
        self.load_image_button = QPushButton("Load Image")
        self.load_image_button.clicked.connect(self.load_image)
        self.control_layout.addWidget(self.load_image_button)

        # add button to apply filters
        self.apply_filter_button = QPushButton("Apply Frequency Filter")
        self.apply_filter_button.clicked.connect(self.apply_filter)
        self.apply_filter_button.setEnabled(False)  # Disabled until image is loaded
        self.control_layout.addWidget(self.apply_filter_button)

        self.control_frame.setLayout(self.control_layout)
        self.layout.addWidget(self.control_frame)

        # Status label
        self.status_label = QLabel("Load an image to begin")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.status_label)

        self.setLayout(self.layout)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            self.modified_image = self.image.copy()
            self.display_image(self.image, self.image_label)
            self.display_image(self.modified_image, self.modified_image_label)

            # Enable the apply filter button
            self.apply_filter_button.setEnabled(True)
            self.status_label.setText("Image loaded. Click 'Apply Frequency Filter' to filter in frequency domain.")

    def display_image(self, image, label):
        qimage = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_Grayscale8)
        pixmap = QPixmap(qimage)
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)

    def apply_filter(self):
        if self.image is None:
            self.status_label.setText("Please load an image first")
            return

        # Create and show the frequency filter dialog
        dialog = FrequencyFilterDialog(self.image, self)
        if dialog.exec_():
            # Get the filter mask from the dialog
            mask = dialog.get_filter_mask()

            # Apply the mask in frequency domain
            filtered_fshift = dialog.fshift * mask

            # Inverse FFT to get back to spatial domain
            f_ishift = np.fft.ifftshift(filtered_fshift)
            filtered_image = np.fft.ifft2(f_ishift)
            filtered_image = np.abs(filtered_image).astype(np.uint8)

            # Update the display
            self.modified_image = filtered_image
            self.display_image(self.modified_image, self.modified_image_label)

            # Update status
            filter_type = "Low-Pass" if dialog.low_pass_radio.isChecked() else "High-Pass"
            self.status_label.setText(f"{filter_type} filter applied successfully")


class FrequencyRegionSelect(QWidget):
    region_changed = Signal(int)  # Signal emits radius

    def __init__(self, parent=None, dialog=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setMouseTracking(True)
        self.parent_dialog = dialog  # Store reference to the parent dialog

        # Initialize with parent size or default if parent is None
        if parent:
            self.setFixedSize(parent.size())
        else:
            self.setFixedSize(400, 400)

        # This is the actual frequency image rectangle (will be updated later)
        self.image_rect = QRect(0, 0, self.width(), self.height())

        # Calculate center of the frequency image
        self.center = QPoint(self.image_rect.width() // 2, self.image_rect.height() // 2)

        # Set initial radius to 1/4 of the smallest dimension
        self.radius = min(self.image_rect.width(), self.image_rect.height()) // 4
        self.min_radius = 10
        self.max_radius = min(self.image_rect.width(), self.image_rect.height()) // 2 - 10

        # Track mouse states
        self.is_dragging = False
        self.last_mouse_pos = None

    def set_image_rect(self, rect):
        """Set the actual image rectangle coordinates within the label"""
        self.image_rect = rect

        # Recalculate center based on actual image position
        self.center = QPoint(
            self.image_rect.x() + self.image_rect.width() // 2,
            self.image_rect.y() + self.image_rect.height() // 2
        )

        # Adjust maximum radius based on the image size
        self.max_radius = min(self.image_rect.width(), self.image_rect.height()) // 2 - 10

        # Ensure current radius is within bounds
        self.radius = min(self.radius, self.max_radius)

        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Check if click is near the circle boundary (within 10 pixels)
            distance = math.sqrt((event.pos().x() - self.center.x()) ** 2 +
                                 (event.pos().y() - self.center.y()) ** 2)

            if abs(distance - self.radius) < 10:
                self.is_dragging = True
                self.last_mouse_pos = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_dragging:
            self.is_dragging = False
            self.region_changed.emit(self.radius)

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            # Calculate the distance from center to current mouse position
            distance = math.sqrt((event.pos().x() - self.center.x()) ** 2 +
                                 (event.pos().y() - self.center.y()) ** 2)

            # Constrain radius within limits
            new_radius = max(self.min_radius, min(distance, self.max_radius))

            if new_radius != self.radius:
                self.radius = new_radius
                self.update()
                self.region_changed.emit(self.radius)
        else:
            # Change cursor to indicate resizing when near the circle boundary
            distance = math.sqrt((event.pos().x() - self.center.x()) ** 2 +
                                 (event.pos().y() - self.center.y()) ** 2)

            if abs(distance - self.radius) < 10:
                self.setCursor(Qt.SizeAllCursor)
            else:
                self.setCursor(Qt.ArrowCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Determine filter type (from parent dialog)
        is_low_pass = True  # Default to low pass
        if hasattr(self.parent().parent(), "low_pass_radio"):
            is_low_pass = self.parent().parent().low_pass_radio.isChecked()

        # Only draw overlay if within the image area
        if self.image_rect.isValid():
            # Create semi-transparent overlay
            if is_low_pass:
                # For low-pass: shade OUTSIDE the circle (blue)
                # First, fill the entire image area with blue overlay
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(0, 0, 255, 80))  # Blue, semi-transparent
                painter.drawRect(self.image_rect)

                # Then clear the circle (make it transparent)
                painter.setCompositionMode(QPainter.CompositionMode_Clear)
                painter.drawEllipse(self.center, self.radius, self.radius)
                painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            else:
                # For high-pass: shade INSIDE the circle (red)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(255, 0, 0, 80))  # Red, semi-transparent
                painter.drawEllipse(self.center, self.radius, self.radius)

            # Draw the circle outline with white pen
            painter.setPen(QPen(Qt.white, 2, Qt.SolidLine))
            painter.setBrush(Qt.NoBrush)  # No fill (transparent inside)
            painter.drawEllipse(self.center, self.radius, self.radius)

            # Draw resize handles on the circle at cardinal points
            handle_size = 6
            handle_points = [
                QPoint(self.center.x(), self.center.y() - self.radius),  # Top
                QPoint(self.center.x() + self.radius, self.center.y()),  # Right
                QPoint(self.center.x(), self.center.y() + self.radius),  # Bottom
                QPoint(self.center.x() - self.radius, self.center.y())  # Left
            ]

            painter.setBrush(QColor(255, 255, 255))
            for point in handle_points:
                painter.drawEllipse(point, handle_size, handle_size)

        painter.end()


class FrequencyFilterDialog(QDialog):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Frequency Domain Filter")
        self.setMinimumSize(800, 600)

        # Store the original image and compute its FFT
        self.image = image
        self.compute_fft()

        # Set up layout
        layout = QVBoxLayout(self)

        # Create radio buttons for filter type
        filter_type_layout = QHBoxLayout()
        self.low_pass_radio = QRadioButton("Low-Pass Filter")
        self.low_pass_radio.setChecked(True)
        self.high_pass_radio = QRadioButton("High-Pass Filter")

        # Group buttons to ensure only one can be selected at a time
        self.filter_buttons = QButtonGroup(self)
        self.filter_buttons.addButton(self.low_pass_radio)
        self.filter_buttons.addButton(self.high_pass_radio)

        # Add buttons to layout
        filter_type_layout.addWidget(self.low_pass_radio)
        filter_type_layout.addWidget(self.high_pass_radio)
        layout.addLayout(filter_type_layout)

        # Add explanation text
        self.explanation_label = QLabel("Select frequency cutoff by dragging the circle boundary. "
                                        "Low-pass filter keeps frequencies inside the circle (passes low frequencies). "
                                        "High-pass filter keeps frequencies outside the circle (passes high frequencies).")
        self.explanation_label.setWordWrap(True)
        self.explanation_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.explanation_label)

        # Create the FFT image display
        self.fft_frame = QFrame()
        self.fft_layout = QVBoxLayout(self.fft_frame)

        # Label for FFT image
        self.fft_label = QLabel("Frequency Domain")
        self.fft_label.setAlignment(Qt.AlignCenter)
        self.fft_layout.addWidget(self.fft_label)

        # Create a container widget for the FFT image and overlay
        self.fft_container = QWidget()
        self.fft_container_layout = QVBoxLayout(self.fft_container)
        self.fft_container_layout.setContentsMargins(0, 0, 0, 0)

        # FFT image display widget
        self.fft_image_label = QLabel()
        self.fft_image_label.setAlignment(Qt.AlignCenter)
        self.fft_container_layout.addWidget(self.fft_image_label)

        # Add FFT container to layout
        self.fft_layout.addWidget(self.fft_container)
        layout.addWidget(self.fft_frame)

        # Create and set up region selection overlay
        self.region_select = FrequencyRegionSelect(self.fft_image_label, self)
        self.region_select.region_changed.connect(self.update_filter_view)

        # Button layout
        button_layout = QHBoxLayout()
        self.apply_button = QPushButton("Apply")
        self.cancel_button = QPushButton("Cancel")
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        # Connect buttons
        self.apply_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        # Connect radio buttons to update the view
        self.low_pass_radio.toggled.connect(self.update_filter_view)
        self.high_pass_radio.toggled.connect(self.update_filter_view)

        # Display the FFT image
        self.display_fft()

    def compute_fft(self):
        # Compute the 2D FFT of the image
        self.f = np.fft.fft2(self.image)
        self.fshift = np.fft.fftshift(self.f)

        # Calculate the magnitude spectrum (log scale for better visualization)
        self.magnitude_spectrum = 20 * np.log(np.abs(self.fshift) + 1)

        # Normalize to 0-255 for display
        self.magnitude_spectrum = self.normalize_image(self.magnitude_spectrum)

    def normalize_image(self, img):
        # Normalize image to 0-255 range
        min_val = np.min(img)
        max_val = np.max(img)
        return ((img - min_val) * 255 / (max_val - min_val)).astype(np.uint8)

    def display_fft(self):
        if not hasattr(self, 'magnitude_spectrum'):
            return

        # Display the FFT magnitude spectrum
        qimage = QImage(self.magnitude_spectrum.data,
                        self.magnitude_spectrum.shape[1],
                        self.magnitude_spectrum.shape[0],
                        self.magnitude_spectrum.strides[0],
                        QImage.Format_Grayscale8)
        pixmap = QPixmap(qimage)

        # Get the label's size
        label_width = self.fft_image_label.width()
        label_height = self.fft_image_label.height()

        # Scale the pixmap to fit the label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(label_width, label_height,
                                      Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation)

        # Calculate the position of the pixmap within the label
        pixmap_x = (label_width - scaled_pixmap.width()) // 2
        pixmap_y = (label_height - scaled_pixmap.height()) // 2

        # Set the pixmap to the label
        self.fft_image_label.setPixmap(scaled_pixmap)

        # Create a rectangle representing the actual image area
        image_rect = QRect(pixmap_x, pixmap_y,
                           scaled_pixmap.width(),
                           scaled_pixmap.height())

        # Update the region select overlay with the actual image position
        if hasattr(self, 'region_select'):
            self.region_select.setFixedSize(self.fft_image_label.size())
            self.region_select.set_image_rect(image_rect)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Re-display the FFT to update positioning when dialog is resized
        self.display_fft()

    def showEvent(self, event):
        super().showEvent(event)
        # Make sure everything is positioned correctly when shown
        self.display_fft()

    def update_filter_view(self):
        # Update the overlay when filter type changes or radius changes
        if hasattr(self, 'region_select'):
            self.region_select.update()

    def get_filter_mask(self):
        # Create a circular mask based on the selected radius and filter type
        rows, cols = self.image.shape
        mask = np.zeros((rows, cols), np.float32)

        # Create a grid of coordinates
        center_y, center_x = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]

        # Calculate distances from center
        dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        # Get the selected radius (scaled to image dimensions)
        if hasattr(self, 'region_select') and hasattr(self.region_select, 'image_rect'):
            # Get the actual image dimensions in the overlay
            image_dim = min(self.region_select.image_rect.width(),
                            self.region_select.image_rect.height())

            # Calculate the scale factor
            scale_factor = min(rows, cols) / image_dim

            # Scale the radius by this factor
            radius = self.region_select.radius * scale_factor
        else:
            # Fallback if image_rect isn't available
            scale_factor = min(rows, cols) / min(self.fft_image_label.width(),
                                                 self.fft_image_label.height())
            radius = self.region_select.radius * scale_factor

        # Create the mask based on filter type
        is_low_pass = self.low_pass_radio.isChecked()

        if is_low_pass:
            # Low-pass filter: keep frequencies inside the circle
            mask[dist_from_center <= radius] = 1.0
        else:
            # High-pass filter: keep frequencies outside the circle
            mask[dist_from_center > radius] = 1.0

        return mask


if __name__ == "__main__":
    app = QApplication([])

    window = QMainWindow()
    window.setCentralWidget(FrequencyFilterWidget())
    window.show()

    sys.exit(app.exec_())