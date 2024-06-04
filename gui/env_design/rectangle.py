import sys
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget
from mazepy.gui.env_design.element import ParameterItem, PlotStandardShape
from mazepy.gui import ENV_SIDE_LENGTH_DEFAULT
from mazepy import clear_spines
import numpy as np

class Rectangle(QWidget):
    def __init__(self):
        super().__init__()
        
        self.side_item_a = ParameterItem("Enter the side a length (unit: cm)")
        self.side_item_a.para_value_spin.valueChanged.connect(self.set_side_a)
        self.side_item_a.para_value_spin.valueChanged.connect(self.visualize)
        
        self.side_item_b = ParameterItem("Enter the side b length (unit: cm)")
        self.side_item_b.para_value_spin.valueChanged.connect(self.set_side_b)
        self.side_item_b.para_value_spin.valueChanged.connect(self.visualize)
        
        side_layout = QHBoxLayout()
        side_layout.addWidget(self.side_item_a)
        side_layout.addWidget(self.side_item_b)
        
        self.a, self.b = ENV_SIDE_LENGTH_DEFAULT, ENV_SIDE_LENGTH_DEFAULT
        
        self.canvas = PlotStandardShape()
        
        layout = QVBoxLayout()
        layout.addLayout(side_layout)
        layout.addWidget(self.canvas)
        
        self.visualize()
        self.setLayout(layout)
        
    def set_side_a(self):
        self.a = int(float(self.side_item_a.para_value_spin.value())*10)/10
        
    def set_side_b(self):
        self.b = int(float(self.side_item_b.para_value_spin.value())*10)/10
        
    def visualize(self):
        x = np.linspace(-self.a*0.5, self.a*0.5, 10000)
        y = np.linspace(-self.b*0.5, self.b*0.5, 10000)
        self.canvas.ax.clear()

        self.ax = clear_spines(self.canvas.ax, set_invisible_spines='all')
        self.ax.set_aspect('equal')

        self.canvas.ax.plot(x, np.repeat(self.b*0.5, 10000), color = 'cornflowerblue')
        self.canvas.ax.plot(x, np.repeat(-self.b*0.5, 10000), color = 'cornflowerblue')
        self.canvas.ax.plot(np.repeat(self.a*0.5, 10000), y, color = 'cornflowerblue')
        self.canvas.ax.plot(np.repeat(-self.a*0.5, 10000), y, color = 'cornflowerblue')
        self.canvas.ax.text(-self.a*0.5, 0, 'b = '+str(self.b), va='center', ha = 'right', rotation = 'vertical')
        self.canvas.ax.text(0, self.b*0.5, 'a = '+str(self.a), ha='center')
        
        self.ax.text(-self.a*0.5, self.b*0.5, 'A', ha = 'right', va = 'bottom')
        self.ax.text(-self.a*0.5,-self.b*0.5, 'B', ha = 'right', va = 'top')
        self.ax.text(self.a*0.5, -self.b*0.5, 'C', ha = 'left',  va = 'top')
        self.ax.text(self.a*0.5,  self.b*0.5, 'D', ha = 'left',  va = 'bottom')
        self.canvas.canvas.draw()
        
        self.identity = {
            'Shape': 'Rectangle',
            'a': self.a, 'b': self.b,            
            'vertex': {
                'A': (0, self.b, 'upper left'), 
                'B': (0, 0, 'bottom left'), 
                'C': (self.a, 0, 'bottom right'), 
                'D': (self.a, self.b, 'upper right')},
            'center': (self.a/2, self.b/2),
            'width': self.a, 
            'height': self.b
        }