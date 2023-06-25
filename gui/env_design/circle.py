import sys
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QDoubleSpinBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mazepy.gui import ParameterItem, ENV_EDGE_LENGTH_DEFAULT
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class Circle(QVBoxLayout):
    def __init__(self, parant: QWidget):
        super().__init__(parent=parant)
        
        self.radius_item = ParameterItem("Enter radius (unit: cm)")
        self.radius_item.para_value_spin.valueChanged.connect(self.set_radius)
        
        self.radius = ENV_EDGE_LENGTH_DEFAULT
        
        self.addWidget(self.radius_item)
        
    def set_radius(self):
        self.radius = float(self.radius_item.para_value_spin.value())
        

        
    
        
        
        
        
        
        