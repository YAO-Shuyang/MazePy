import sys
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QDoubleSpinBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mazepy.gui.env_design.element import ParameterItem, PlotStandardShape
from mazepy.gui import ENV_EDGE_LENGTH_DEFAULT
from mazepy import clear_spines
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

class Circle(QWidget):
    def __init__(self):
        super().__init__()
        
        self.radius_item = ParameterItem("Enter radius (unit: cm)")
        self.radius_item.para_value_spin.valueChanged.connect(self.set_radius)
        self.radius_item.para_value_spin.valueChanged.connect(self.visualize)
        
        self.radius = ENV_EDGE_LENGTH_DEFAULT
        
        self.canvas = PlotStandardShape()
        
        layout = QVBoxLayout()
        layout.addWidget(self.radius_item)
        layout.addWidget(self.canvas)
        
        self.visualize()
        self.setLayout(layout)
        
    def set_radius(self):
        self.radius = int(float(self.radius_item.para_value_spin.value())*10)/10
        
    def visualize(self):
        theta = np.linspace(0,360, 10000)
        self.canvas.ax.clear()

        self.ax = clear_spines(self.canvas.ax, set_invisible_spines='plot')
        self.ax.spines['left'].set_position('zero')
        self.ax.spines['bottom'].set_position('zero')
        self.ax.set_xticks([])
        self.ax.set_yticks([])        

        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)
        self.canvas.ax.plot(x, y, color = 'cornflowerblue')
        self.canvas.ax.text(self.radius*0.5, 0, str(self.radius), ha='center')
        self.canvas.canvas.draw()
        

        
    
        
        
        
        
        
        