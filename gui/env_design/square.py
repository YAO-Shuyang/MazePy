import sys
from PyQt6.QtWidgets import QVBoxLayout, QWidget
from mazepy.gui.env_design.element import ParameterItem, PlotStandardShape
from mazepy.gui import ENV_SIDE_LENGTH_DEFAULT
from mazepy import clear_spines
import numpy as np

class Square(QWidget):
    def __init__(self):
        super().__init__()
        
        self.side_item = ParameterItem("Enter the side length (unit: cm)")
        self.side_item.para_value_spin.valueChanged.connect(self.set_side)
        self.side_item.para_value_spin.valueChanged.connect(self.visualize)
        
        self.side = ENV_SIDE_LENGTH_DEFAULT
        
        self.canvas = PlotStandardShape()
        
        layout = QVBoxLayout()
        layout.addWidget(self.side_item)
        layout.addWidget(self.canvas)
        
        self.visualize()
        self.setLayout(layout)
    
    def set_side(self):
        self.side = int(float(self.side_item.para_value_spin.value())*10)/10
        
    def visualize(self):
        x = np.linspace(-self.side*0.5, self.side*0.5, 10000)
        y = np.linspace(-self.side*0.5, self.side*0.5, 10000)
        self.canvas.ax.clear()

        self.ax = clear_spines(self.canvas.ax, set_invisible_spines='all')   
        self.ax.set_aspect('equal')     

        self.ax.plot(x, np.repeat(self.side*0.5, 10000), color = 'cornflowerblue')
        self.ax.plot(x, np.repeat(-self.side*0.5, 10000), color = 'cornflowerblue')
        self.ax.plot(np.repeat(self.side*0.5, 10000), y, color = 'cornflowerblue')
        self.ax.plot(np.repeat(-self.side*0.5, 10000), y, color = 'cornflowerblue')
        self.ax.text(-self.side*0.5, 0, str(self.side), va='center', ha = 'right', rotation = 'vertical')
        self.ax.text(0, self.side*0.5, str(self.side), ha='center')
        
        self.ax.text(-self.side*0.5, self.side*0.5, 'A', ha = 'right', va = 'bottom')
        self.ax.text(-self.side*0.5,-self.side*0.5, 'B', ha = 'right', va = 'top')
        self.ax.text(self.side*0.5, -self.side*0.5, 'C', ha = 'left',  va = 'top')
        self.ax.text(self.side*0.5,  self.side*0.5, 'D', ha = 'left',  va = 'bottom')
        self.canvas.canvas.draw()
        
        self.identity = {
            'Shape': 'Square',
            'a': self.side,
            'vertex': {
                'A': (0, self.side, 'upper left'), 
                'B': (0, 0, 'bottom left'), 
                'C': (self.side, 0, 'bottom right'), 
                'D': (self.side, self.side, 'upper right')},
            'center': (self.side/2, self.side/2),
            'width': self.side, 
            'height': self.side
        }