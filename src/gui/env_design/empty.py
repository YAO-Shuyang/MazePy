from PyQt6.QtWidgets import QVBoxLayout, QWidget
from mazepy.gui.env_design.element import ParameterItem, PlotStandardShape
from mazepy import clear_spines

class Empty(QWidget):
    def __init__(self):
        super().__init__()
        
        self.empty_item = ParameterItem("Input locked")
        self.empty_item.para_value_spin.setRange(0, 0)
        self.empty_item.para_value_spin.setValue(0)
        self.empty_item.para_value_spin.valueChanged.connect(self.set_empty)
        
        self.empty = 0
        
        self.canvas = PlotStandardShape()
        
        layout = QVBoxLayout()
        layout.addWidget(self.empty_item)
        layout.addWidget(self.canvas)
        
        self.visualize()
        self.setLayout(layout)
        
    def set_empty(self):
        self.empty_item.para_value_spin.setValue(0)
        
    def visualize(self):
        self.canvas.ax.clear()

        self.ax = clear_spines(self.canvas.ax, set_invisible_spines='plot')
        self.ax.spines['left'].set_position('zero')
        self.ax.spines['bottom'].set_position('zero')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.axis([-1,1,-1,1]) 

        self.canvas.canvas.draw()
        
        self.identity = {}