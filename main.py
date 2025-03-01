from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile
from equalizing import EqualizingWidget
from thresholding import ThresholdingWidget
from addNoise import AddNoiseWidget
from filterNoise import FilterNoiseWidget
from detectEdges import DetectEdgesWidget
from frequencyFilter import FrequencyFilterWidget
from hybridImages import HybridImagesWidget
import sys


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loader = QUiLoader()
        ui_file = QFile("main.ui")
        ui_file.open(QFile.ReadOnly)
        self.ui = loader.load(ui_file, self)
        ui_file.close()
        
        self.tabWidget = self.ui.findChild(QTabWidget, "main")

        
        self.equalizing_widget = EqualizingWidget()
        self.thresholding_widget = ThresholdingWidget()
        self.addNoise_widget = AddNoiseWidget()
        self.filterNoise_widget = FilterNoiseWidget()
        self.detectEdges_widget = DetectEdgesWidget()
        self.frequencyFilter_widget = FrequencyFilterWidget()
        self.hybridImages_widget = HybridImagesWidget()

        filters_tab = self.tabWidget.widget(0)
        frequency_domain_filters_tab = self.tabWidget.widget(1)
        noise_tab = self.tabWidget.widget(2)
        edge_detection_tab = self.tabWidget.widget(3)
        equalization_tab = self.tabWidget.widget(4)
        thresholding_tab = self.tabWidget.widget(6)
        hybrid_tab = self.tabWidget.widget(7)

        # Ensure the tabs have layouts to add widgets
        if not equalization_tab.layout():
            from PySide6.QtWidgets import QVBoxLayout
            equalization_tab.setLayout(QVBoxLayout())

        if not frequency_domain_filters_tab.layout():
            from PySide6.QtWidgets import QVBoxLayout
            frequency_domain_filters_tab.setLayout(QVBoxLayout())
        
        if not thresholding_tab.layout():
            from PySide6.QtWidgets import QVBoxLayout
            thresholding_tab.setLayout(QVBoxLayout())
        
        if not filters_tab.layout():
            from PySide6.QtWidgets import QVBoxLayout
            filters_tab.setLayout(QVBoxLayout())
        
        if not noise_tab.layout():
            from PySide6.QtWidgets import QVBoxLayout
            noise_tab.setLayout(QVBoxLayout())
        
        if not edge_detection_tab.layout():
            from PySide6.QtWidgets import QVBoxLayout
            edge_detection_tab.setLayout(QVBoxLayout())

        if not hybrid_tab.layout():
            from PySide6.QtWidgets import QVBoxLayout
            hybrid_tab.setLayout(QVBoxLayout())

        equalization_tab.layout().addWidget(self.equalizing_widget)
        frequency_domain_filters_tab.layout().addWidget(self.frequencyFilter_widget)
        thresholding_tab.layout().addWidget(self.thresholding_widget)
        noise_tab.layout().addWidget(self.addNoise_widget)
        filters_tab.layout().addWidget(self.filterNoise_widget)
        edge_detection_tab.layout().addWidget(self.detectEdges_widget)
        hybrid_tab.layout().addWidget(self.hybridImages_widget)

        self.setCentralWidget(self.ui.findChild(QWidget, "centralwidget"))


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    window.showMaximized()
    sys.exit(app.exec())
