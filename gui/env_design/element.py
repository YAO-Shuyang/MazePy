import typing
from PyQt6 import QtCore
from PyQt6.QtWidgets import QApplication, QComboBox, QCheckBox, QPushButton, QFileDialog, QHBoxLayout, QScrollArea
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QLineEdit, QWidget, QDoubleSpinBox
from mazepy.gui import WarningWindow, NoticeWindow
from mazepy import mkdir
from mazepy.gui import ENV_OUTSIDE_SHAPE, ENV_EDGE_LENGTH_DEFAULT, LARGEST_LEMGTJ_DEFAULT, SMALLEST_LEMGTJ_DEFAULT
import yaml
import sys
import os

class ParameterItem(QWidget):
    def __init__(self, label: str, ) -> None:
        super().__init__()
        
        # input radius
        self.para_label = QLabel(label)
        self.para_label.setWordWrap(True)
        self.para_value_spin = QDoubleSpinBox()
        self.para_value_spin.setSingleStep(0.1)
        self.para_value_spin.valueChanged.connect(self.set_value)
        self.para_value_spin.setRange(SMALLEST_LEMGTJ_DEFAULT, LARGEST_LEMGTJ_DEFAULT)
        self.para_value = ENV_EDGE_LENGTH_DEFAULT
        self.para_value_spin.setValue(ENV_EDGE_LENGTH_DEFAULT)
        
        layout = QVBoxLayout()
        layout.addWidget(self.para_label)
        layout.addWidget(self.para_value_spin)
        
        self.setLayout(layout)
        
    def set_value(self):
        self.para_value = float(self.para_value.text())
        

class PlotStandardShape(QWidget):
    def __init__(self, label: str, ) -> None:
        super().__init__()    