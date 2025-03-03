import sys
import cv2
import numpy as np
from PySide6.QtCore import QFile
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QPushButton, QWidget
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtUiTools import QUiLoader
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class NormalizationWidget(QWidget):
    def __init__(self):
        super().__init__()
        loader = QUiLoader()
        ui_file = QFile("normalization_histogram.ui")
        ui_file.open(QFile.ReadOnly)
        self.ui = loader.load(ui_file, self)
        ui_file.close()

        self.LoadImageBtn = self.ui.findChild(QPushButton, "LoadImage")
        self.NormalizeBtn = self.ui.findChild(QPushButton, "Normlize")

        self.LoadImageBtn.clicked.connect(self.load_image)
        self.NormalizeBtn.clicked.connect(self.normalize_image)
        self.image = None
        self.normalized_image = None

    def load_image(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
            if file_path:
                
                self.clear_widget(self.InputImage)
                self.clear_widget(self.OutputImage)
            
                self.image = cv2.imread(file_path)
                if self.image is None:
                    print("Error: Failed to load image.")
                    return

                self.display_image(self.image, self.InputImage)

               
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
            label.setPixmap(pixmap)
            label.setScaledContents(True)

           
            label.setFixedSize(831, 452)  

            # Apply a border radius and other styles using a style sheet
            label.setStyleSheet("""
                QLabel {
                    background-color: #F2EEEB;  
                    border: 4px solid #3E2B22;  
                    border-radius: 20px;  
                    padding: 10px;  
                }
            """)

            # Set the QLabel as the only widget in the layout
            layout = widget.layout()
            if layout is None:
                layout = QVBoxLayout(widget)
                layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
                widget.setLayout(layout)
            else:
                # Clear the existing layout
                for i in reversed(range(layout.count())):
                    layout.itemAt(i).widget().setParent(None)

            # Add the QLabel to the layout
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
                        layout.itemAt(i).widget().setParent(None)
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
        cdf_normalized = cdf / cdf.max()  # Normalize to the range [0, 1]
        return cdf_normalized

    def display_histograms(self, image):
        try:
            # Calculate histograms for each channel
            if len(image.shape) == 2:  # Grayscale image
                histogram = self.calculate_histogram(image)
                cdf = self.calculate_cdf(histogram)
                self.plot_histogram([histogram], ['black'], self.Histogram, is_cdf=False)
                self.plot_histogram([cdf], ['red'], self.HistogramCDF, is_cdf=True)
            else:  # RGB image
                red_histogram, green_histogram, blue_histogram = self.calculate_histogram(image)
                red_cdf = self.calculate_cdf(red_histogram)
                green_cdf = self.calculate_cdf(green_histogram)
                blue_cdf = self.calculate_cdf(blue_histogram)

                # Plot histograms and CDFs for each channel
                self.plot_histogram([red_histogram, green_histogram, blue_histogram], ['red', 'green', 'blue'], self.Histogram, is_cdf=False)
                self.plot_histogram([red_cdf, green_cdf, blue_cdf], ['red', 'green', 'blue'], self.HistogramCDF, is_cdf=True)
        except Exception as e:
            print(f"Error in display_histograms: {e}")

    def plot_histogram(self, data_list, colors, widget, is_cdf=False):
        try:
            # Create a matplotlib figure and canvas
            fig = Figure()
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            for data, color in zip(data_list, colors):
                ax.plot(data, color=color)
            ax.set_xlim([0, 256])
            ax.set_facecolor('#D7CBBD')  # Set background color of the plot
            fig.set_facecolor('#F2EEEB')  # Set background color of the figure
            fig.tight_layout()

            # Set the size of the canvas (plot)
            canvas.setFixedSize(800, 244)  # Adjust the size as needed

            # Apply a border radius and other styles to the widget using a style sheet
            widget.setStyleSheet("""
                QWidget {
                    background-color: #F2EEEB;  
                    border: 4px solid #3E2B22;  
                    border-radius: 20px;  
                }
            """)

            # Embed the plot in the widget
            if widget.layout() is None:
                widget.setLayout(QVBoxLayout())
                widget.layout().setContentsMargins(10, 10, 10, 10)  # Add margins
            layout = widget.layout()
            for i in reversed(range(layout.count())):
                layout.itemAt(i).widget().setParent(None)
            layout.addWidget(canvas)
        except Exception as e:
            print(f"Error in plot_histogram: {e}")


    def min_max_normalization(self, image, new_min=0, new_max=255):
        
        
        I_min = np.min(image)
        I_max = np.max(image)

        
        if I_max == I_min:
            return image

        
        normalized_image = (image - I_min) / (I_max - I_min) * (new_max - new_min) + new_min

        
        normalized_image = np.uint8(normalized_image)

        return normalized_image

    def normalize_image(self):
        try:
            if self.image is not None:
                # Perform min-max normalization manually
                self.normalized_image = self.min_max_normalization(self.image)

                # Clear the output widget before displaying the normalized image
                self.clear_widget(self.OutputImage)

                # Display the normalized image
                self.display_image(self.normalized_image, self.OutputImage)
        except Exception as e:
            print(f"Error in normalize_image: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = QMainWindow()
    window.setCentralWidget(NormalizationWidget())
    window.show()

    sys.exit(app.exec_())
