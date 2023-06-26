from PyQt6.QtWidgets import QVBoxLayout, QWidget
from mazepy.gui.env_design.element import ParameterItem, PlotStandardShape
from mazepy.gui import ENV_SIDE_LENGTH_DEFAULT
from mazepy import clear_spines
import numpy as np

class Circle(QWidget):
    def __init__(self):
        super().__init__()
        
        self.radius_item = ParameterItem("Enter the radius (unit: cm)")
        self.radius_item.para_value_spin.valueChanged.connect(self.set_radius)
        self.radius_item.para_value_spin.valueChanged.connect(self.visualize)
        
        self.radius = ENV_SIDE_LENGTH_DEFAULT
        
        self.canvas = PlotStandardShape()
        
        layout = QVBoxLayout()
        layout.addWidget(self.radius_item)
        layout.addWidget(self.canvas)
        
        self.visualize()
        self.setLayout(layout)
        
    def set_radius(self):
        self.radius = int(float(self.radius_item.para_value_spin.value())*10)/10
        
    def visualize(self):
        theta = np.linspace(0,2*np.pi, 10000)
        self.canvas.ax.clear()

        self.ax = clear_spines(self.canvas.ax, set_invisible_spines='all')
        self.ax.set_aspect('equal')   

        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)
        self.ax.plot(x, y, color = 'cornflowerblue')
        self.ax.text(self.radius*0.5, 0, 'r = '+str(self.radius), ha='center', rotation = -5)
        self.ax.arrow(0, 0, self.radius*np.cos(np.pi/36)*0.95, -self.radius*np.sin(np.pi/36), color = 'black', head_width = 5, head_length = 5)
        self.canvas.canvas.draw()
        
        self.identity = {
            'Shape': 'Circle',
            'radius': self.radius,
            'center': (self.radius, self.radius),
            'width': 2*self.radius, 
            'height': 2*self.radius
        }