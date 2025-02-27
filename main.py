from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile
from equalizing import EqualizingWidget
from thresholding import ThresholdingWidget
from addNoise import AddNoiseWidget
from filterNoise import FilterNoiseWidget
from detectEdges import DetectEdgesWidget
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

        filters_tab = self.tabWidget.widget(0)
        noise_tab = self.tabWidget.widget(1)
        edge_detection_tab = self.tabWidget.widget(2)
        equalization_tab = self.tabWidget.widget(3)
        thresholding_tab = self.tabWidget.widget(5)

        # Ensure the tabs have layouts to add widgets
        if not equalization_tab.layout():
            from PySide6.QtWidgets import QVBoxLayout
            equalization_tab.setLayout(QVBoxLayout())
        
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

        equalization_tab.layout().addWidget(self.equalizing_widget)
        thresholding_tab.layout().addWidget(self.thresholding_widget)
        noise_tab.layout().addWidget(self.addNoise_widget)
        filters_tab.layout().addWidget(self.filterNoise_widget)
        edge_detection_tab.layout().addWidget(self.detectEdges_widget)

        self.setCentralWidget(self.ui.findChild(QWidget, "centralwidget"))


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    window.showMaximized()
    sys.exit(app.exec())
