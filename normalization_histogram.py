import sys
import cv2
import numpy as np
from PySide6.QtCore import QFile, Qt, QSize
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QPushButton, QWidget, \
    QHBoxLayout, QSizePolicy, QScrollArea
from PySide6.QtGui import QImage, QPixmap, QScreen
from PySide6.QtUiTools import QUiLoader
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class NormalizationWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.normalized_image = None
        self.initUI()

    def initUI(self):
        # Main layout
        self.mainLayout = QVBoxLayout(self)
        self.mainLayout.setSpacing(10)
        self.mainLayout.setContentsMargins(10, 10, 10, 10)

        # Get available screen size
        screen = QApplication.primaryScreen()
        screen_size = screen.availableSize()

        # Default image display size - more conservative
        self.img_width = min(400, int(screen_size.width() * 0.3))
        self.img_height = min(300, int(screen_size.height() * 0.3))

        # Top section - Input/Output Images
        self.topWidget = QWidget()
        self.topLayout = QHBoxLayout(self.topWidget)
        self.topLayout.setSpacing(10)
        self.topLayout.setContentsMargins(0, 0, 0, 0)  # Remove margins to maximize space

        # Input image section
        self.inputImageWidget = QWidget()
        self.inputImageLayout = QVBoxLayout(self.inputImageWidget)
        self.inputImageLayout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        self.inputImage = QWidget()
        self.inputImageLayout.addWidget(self.inputImage)
        self.inputImage.setObjectName("InputImage")
        self.inputImageWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Buttons section - make it narrower
        self.buttonsWidget = QWidget()
        self.buttonsLayout = QVBoxLayout(self.buttonsWidget)
        self.buttonsLayout.setContentsMargins(5, 0, 5, 0)  # Smaller margins
        self.loadImageBtn = QPushButton("Load Image")
        self.loadImageBtn.setObjectName("LoadImage")
        self.normalizeBtn = QPushButton("Normalize")
        self.normalizeBtn.setObjectName("Normlize")  # Keep original spelling as in the UI file

        self.buttonsLayout.addWidget(self.loadImageBtn)
        self.buttonsLayout.addWidget(self.normalizeBtn)
        self.buttonsLayout.addStretch()

        # Set a fixed width for the buttons column to be narrower
        self.buttonsWidget.setFixedWidth(190)
        self.buttonsWidget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

        # Output image section
        self.outputImageWidget = QWidget()
        self.outputImageLayout = QVBoxLayout(self.outputImageWidget)
        self.outputImageLayout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        self.outputImage = QWidget()
        self.outputImageLayout.addWidget(self.outputImage)
        self.outputImage.setObjectName("OutputImage")
        self.outputImageWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Add all sections to top layout with appropriate stretch factors
        self.topLayout.addWidget(self.inputImageWidget, 1)  # 1 part stretch
        self.topLayout.addWidget(self.buttonsWidget, 0)  # 0 part stretch (fixed width)
        self.topLayout.addWidget(self.outputImageWidget, 1)  # 1 part stretch

        # Bottom section - Histograms
        self.bottomWidget = QWidget()
        self.bottomLayout = QHBoxLayout(self.bottomWidget)
        self.bottomLayout.setSpacing(10)
        self.bottomLayout.setContentsMargins(0, 0, 0, 0)  # Remove margins

        # Histogram section
        self.histogramWidget = QWidget()
        self.histogramLayout = QVBoxLayout(self.histogramWidget)
        self.histogramLayout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        self.histogramLabel = QLabel("Histogram")
        self.histogramLabel.setObjectName("label")
        self.histogram = QWidget()
        self.histogram.setObjectName("Histogram")
        self.histogramLayout.addWidget(self.histogramLabel)
        self.histogramLayout.addWidget(self.histogram)
        self.histogramWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # CDF section
        self.cdfWidget = QWidget()
        self.cdfLayout = QVBoxLayout(self.cdfWidget)
        self.cdfLayout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        self.cdfLabel = QLabel("CDF")
        self.cdfLabel.setObjectName("label")
        self.histogramCDF = QWidget()
        self.histogramCDF.setObjectName("HistogramCDF")
        self.cdfLayout.addWidget(self.cdfLabel)
        self.cdfLayout.addWidget(self.histogramCDF)
        self.cdfWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # Add histogram and CDF to bottom layout
        self.bottomLayout.addWidget(self.histogramWidget, 1)
        self.bottomLayout.addWidget(self.cdfWidget, 1)

        # Add top and bottom sections to main layout
        self.mainLayout.addWidget(self.topWidget)
        self.mainLayout.addWidget(self.bottomWidget)

        # Set reasonable size ratio between top and bottom widgets
        self.mainLayout.setStretch(0, 3)  # Top section gets 3 parts
        self.mainLayout.setStretch(1, 2)  # Bottom section gets 2 parts

        # Connect signals
        self.loadImageBtn.clicked.connect(self.load_image)
        self.normalizeBtn.clicked.connect(self.normalize_image)

        # No maximum size on the main widget - let it expand to fill available space
        self.setMinimumSize(800, 600)

    def load_image(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
            if file_path:
                self.clear_widget(self.inputImage)
                self.clear_widget(self.outputImage)

                self.image = cv2.imread(file_path)
                if self.image is None:
                    print("Error: Failed to load image.")
                    return

                self.display_image(self.image, self.inputImage)
                self.display_histograms(self.image)
        except Exception as e:
            print(f"Error in load_image: {e}")

    def display_image(self, image, widget):
        try:
            # Convert the image to QImage format
            if len(image.shape) == 2:  # Grayscale image
                qimage = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_Grayscale8)
            else:  # RGB image
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qimage = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Create a QLabel to display the image
            label = QLabel(widget)
            pixmap = QPixmap.fromImage(qimage)

            # Scale the pixmap to fit within our target dimensions while maintaining aspect ratio
            if pixmap.width() > 0 and pixmap.height() > 0:
                pixmap = pixmap.scaled(
                    self.img_width,
                    self.img_height,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )

            # Set the pixmap to the label
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)
            # Set size policy to expand and fill available space
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            label.setMinimumSize(self.img_width, self.img_height)

            # Set the label as the only widget in the layout
            layout = widget.layout()
            if layout is None:
                layout = QVBoxLayout(widget)
                layout.setContentsMargins(0, 0, 0, 0)
                widget.setLayout(layout)
            else:
                # Clear the existing layout
                for i in reversed(range(layout.count())):
                    layout.itemAt(i).widget().setParent(None)

            # Add the label to the layout
            layout.addWidget(label)

        except Exception as e:
            print(f"Error in display_image: {e}")

    def clear_widget(self, widget):
        try:
            # Clear the widget by removing its content
            if isinstance(widget, QLabel):
                widget.clear()  # Clear the QLabel
            else:
                # If the widget is a QWidget, clear its layout
                layout = widget.layout()
                if layout is not None:
                    for i in reversed(range(layout.count())):
                        item = layout.itemAt(i)
                        if item.widget():
                            item.widget().setParent(None)
        except Exception as e:
            print(f"Error in clear_widget: {e}")

    def calculate_histogram(self, image):
        if len(image.shape) == 2:  # Grayscale image
            histogram = np.zeros(256, dtype=int)
            for pixel_value in image.ravel():
                histogram[pixel_value] += 1
            return histogram
        else:  # RGB image
            # Split the image into its channels
            b, g, r = cv2.split(image)
            # Calculate histograms for each channel
            red_histogram = self.calculate_histogram(r)
            green_histogram = self.calculate_histogram(g)
            blue_histogram = self.calculate_histogram(b)
            return red_histogram, green_histogram, blue_histogram

    def calculate_cdf(self, histogram):
        cdf = histogram.cumsum()  # Compute the cumulative sum
        cdf_normalized = cdf / cdf.max() if cdf.max() > 0 else cdf  # Normalize to the range [0, 1]
        return cdf_normalized

    def display_histograms(self, image):
        try:
            # Calculate histograms for each channel
            if len(image.shape) == 2:  # Grayscale image
                histogram = self.calculate_histogram(image)
                cdf = self.calculate_cdf(histogram)
                self.plot_histogram([histogram], ['black'], self.histogram, is_cdf=False)
                self.plot_histogram([cdf], ['red'], self.histogramCDF, is_cdf=True)
            else:  # RGB image
                red_histogram, green_histogram, blue_histogram = self.calculate_histogram(image)
                red_cdf = self.calculate_cdf(red_histogram)
                green_cdf = self.calculate_cdf(green_histogram)
                blue_cdf = self.calculate_cdf(blue_histogram)

                # Plot histograms and CDFs for each channel
                self.plot_histogram([red_histogram, green_histogram, blue_histogram], ['red', 'green', 'blue'],
                                    self.histogram, is_cdf=False)
                self.plot_histogram([red_cdf, green_cdf, blue_cdf], ['red', 'green', 'blue'], self.histogramCDF,
                                    is_cdf=True)
        except Exception as e:
            print(f"Error in display_histograms: {e}")

    def plot_histogram(self, data_list, colors, widget, is_cdf=False):
        try:
            # Create a matplotlib figure and canvas with a dynamic size
            fig = Figure(figsize=(4, 3), dpi=100)  # Smaller figure size
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            for data, color in zip(data_list, colors):
                ax.plot(data, color=color)

            ax.set_xlim([0, 256])
            if not is_cdf:
                ax.set_ylim(bottom=0)  # Only set bottom limit for histogram

            ax.set_facecolor('#D7CBBD')  # Set background color of the plot
            fig.set_facecolor('#F2EEEB')  # Set background color of the figure
            fig.tight_layout()

            # Set size policy to allow canvas to expand and fill space
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            # Embed the plot in the widget
            if widget.layout() is None:
                widget.setLayout(QVBoxLayout())
                widget.layout().setContentsMargins(10, 10, 10, 10)

            layout = widget.layout()
            for i in reversed(range(layout.count())):
                layout.itemAt(i).widget().setParent(None)

            layout.addWidget(canvas)
        except Exception as e:
            print(f"Error in plot_histogram: {e}")

    def min_max_normalization(self, image, new_min=0, new_max=255):
        I_min = np.min(image)
        I_max = np.max(image)

        # Avoid division by zero
        if I_max == I_min:
            return image

        # Apply min-max normalization
        normalized_image = (image - I_min) / (I_max - I_min) * (new_max - new_min) + new_min

        # Convert back to uint8
        normalized_image = np.uint8(normalized_image)

        return normalized_image

    def normalize_image(self):
        try:
            if self.image is not None:
                # Perform min-max normalization manually
                self.normalized_image = self.min_max_normalization(self.image)

                # Clear the output widget before displaying the normalized image
                self.clear_widget(self.outputImage)

                # Display the normalized image
                self.display_image(self.normalized_image, self.outputImage)
        except Exception as e:
            print(f"Error in normalize_image: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Create and set up the main window
    window = QMainWindow()
    window.setWindowTitle("Image Normalization")

    # Create the widget
    normalization_widget = NormalizationWidget()
    window.setCentralWidget(normalization_widget)

    window.show()
    sys.exit(app.exec_())