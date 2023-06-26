import sys
from PyQt6.QtWidgets import QVBoxLayout, QWidget, QLabel, QHBoxLayout, QComboBox
from mazepy.gui.env_design.element import ParameterItem, PlotStandardShape, LockedItem
from mazepy.gui import ENV_SIDE_LENGTH_DEFAULT, ENV_ANGLE_DEGREE_DEFAULT
from mazepy import clear_spines
import numpy as np
import copy as cp

class Triangle(QWidget):
    def __init__(self):
        super().__init__()
        
        self.a, self.b, self.c = ENV_SIDE_LENGTH_DEFAULT, ENV_SIDE_LENGTH_DEFAULT, ENV_SIDE_LENGTH_DEFAULT
        self.ab, self.bc, self.ac = ENV_ANGLE_DEGREE_DEFAULT, ENV_ANGLE_DEGREE_DEFAULT, ENV_ANGLE_DEGREE_DEFAULT
        
        select_mode_label = QLabel("Select a mode to ensure a triangle:")
        self.select_mode = QComboBox()
        for mode in ['SSS', 'SAS', 'ASA', 'HL']:
            self.select_mode.addItem(mode)
        self.select_mode.currentTextChanged.connect(self.display_parameters)
        
        self.dynamic_widget = QWidget()
        
        canvas_label = QLabel("Clockwise arrangement of a, b, c")
        self.canvas = PlotStandardShape()
        
        self.layouts = QVBoxLayout()
        self.layouts.addWidget(select_mode_label)
        self.layouts.addWidget(self.select_mode)
        self.layouts.addWidget(self.dynamic_widget)
        self.layouts.addWidget(self.canvas)

        self.setLayout(self.layouts)

    def get_label_info(self):
        a, b, c = self.a, self.b, self.c
        if a >= b and a >= c:
            return {'bottom side': a,
                    'left side': b,
                    'right side': c,
                    'left angle': self.ab,
                    'right angle': self.ac,
                    'bottom label': 'a',
                    'left label': 'b',
                    'right label': 'c'}
        elif b > a and b >= c:
            return {'bottom side': b,
                    'left side': c,
                    'right side': a,
                    'left angle': self.bc,
                    'right angle': self.ab,
                    'bottom label': 'b',
                    'left label': 'c',
                    'right label': 'a'}
        elif c > a and c > b:
            return {'bottom side': c,
                    'left side': a,
                    'right side': b,
                    'left angle': self.ac,
                    'right angle': self.bc,
                    'bottom label': 'c',
                    'left label': 'a',
                    'right label': 'b'}
    
    def display_parameters(self):
        self.layouts.removeWidget(self.dynamic_widget)
        
        if self.select_mode.currentText() == 'SAS':
            print("Change to SAS")
            self.dynamic_widget = SAS()
            self.dynamic_widget.a_item.para_value_spin.valueChanged.connect(self.pass_values)
            self.dynamic_widget.b_item.para_value_spin.valueChanged.connect(self.pass_values)
            self.dynamic_widget.ab_item.para_value_spin.valueChanged.connect(self.pass_values)
        else:
            self.dynamic_widget = QWidget()
            
        self.layouts.insertWidget(self.layouts.count()-1, self.dynamic_widget)
        
    def visualize(self):
        label_info = self.get_label_info()
        
        x_l, x_r = -label_info['bottom side']*0.5, label_info['bottom side']*0.5
        x_t = x_l + label_info['left side']*np.cos(label_info['left angle'] * np.pi / 180)
        y_t = label_info['left side']*np.sin(label_info['left angle'] * np.pi / 180)
        
        self.canvas.ax.clear()

        self.ax = clear_spines(self.canvas.ax, set_invisible_spines='all')
        self.ax.set_aspect('equal')
        
        self.ax.plot([x_l, x_r], [0, 0], color = 'cornflowerblue')
        self.ax.plot([x_l, x_t], [0, y_t], color = 'cornflowerblue')
        self.ax.plot([x_r, x_t], [0, y_t], color = 'cornflowerblue')
        
        self.ax.text(0, 0, label_info['bottom label']+' = '+str(round(label_info['bottom side'], 2)), ha = 'center', va = 'top')
        self.ax.text((x_l+x_t)/2, y_t/2, label_info['left label']+' = '+str(round(label_info['left side'], 2)), ha = 'right', va = 'center', rotation = label_info['left angle'])
        self.ax.text((x_r+x_t)/2, y_t/2, label_info['right label']+' = '+str(round(label_info['right side'], 2)), ha = 'left', va = 'center', rotation = -label_info['right angle'])
        self.canvas.canvas.draw()

    def pass_values(self):
        self.a, self.b, self.c = self.dynamic_widget.a, self.dynamic_widget.b, self.dynamic_widget.c
        self.ab, self.bc, self.ac = self.dynamic_widget.ab, self.dynamic_widget.bc, self.dynamic_widget.ac
        self.visualize()

class SAS(QWidget):
    def __init__(self):
        super().__init__()
        
        self.a, self.b, self.ab = ENV_SIDE_LENGTH_DEFAULT, ENV_SIDE_LENGTH_DEFAULT, ENV_ANGLE_DEGREE_DEFAULT
        
        input_label = QLabel("SAS Mode: using two side length (a, b, unit: cm) and there included angle (∠ab, unit: degree) to ensure a triangle.")
        input_label.setWordWrap(True)
        self.a_item = ParameterItem("Enter a")
        self.a_item.para_value_spin.setValue(ENV_SIDE_LENGTH_DEFAULT)
        self.b_item = ParameterItem("Enter b")
        self.b_item.para_value_spin.setValue(ENV_SIDE_LENGTH_DEFAULT)
        self.ab_item = ParameterItem("Enter ∠ab")
        self.ab_item.para_value_spin.setRange(0.01, 179.99)
        self.ab_item.para_value_spin.setValue(ENV_ANGLE_DEGREE_DEFAULT)
        
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.a_item, 1)
        input_layout.addWidget(self.b_item, 1)
        input_layout.addWidget(self.ab_item, 1)
        
        output_label = QLabel("Related values")
        output_label.setWordWrap(True)
        self.c_item = LockedItem("c")
        self.bc_item = LockedItem("Angle ∠bc")
        self.ac_item = LockedItem("Angle ∠ac")
        
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.c_item, 1)
        output_layout.addWidget(self.bc_item, 1)
        output_layout.addWidget(self.ac_item, 1)
        
        
        values_layout = QVBoxLayout()
        values_layout.addWidget(input_label)
        values_layout.addLayout(input_layout)
        values_layout.addWidget(output_label)
        values_layout.addLayout(output_layout)
        
        self.a_item.para_value_spin.valueChanged.connect(self.set_a)
        self.a_item.para_value_spin.valueChanged.connect(self.calc_values)

        self.b_item.para_value_spin.valueChanged.connect(self.set_b)
        self.b_item.para_value_spin.valueChanged.connect(self.calc_values)
        
        self.ab_item.para_value_spin.valueChanged.connect(self.set_ab)
        self.ab_item.para_value_spin.valueChanged.connect(self.calc_values)
        
        self.setLayout(values_layout)
        
        self.calc_values()
    
    def set_a(self):
        self.a = float(self.a_item.para_value_spin.text())

    def set_b(self):
        self.b = float(self.b_item.para_value_spin.text())
        
    def set_ab(self):
        self.ab = float(self.ab_item.para_value_spin.text())
    
    def calc_values(self):
        a, b, ab = self.a, self.b, self.ab
        c = np.sqrt(a**2 + b**2 - 2*a*b*np.cos(ab*np.pi/180))
        
        if a**2 + c**2 == b**2:
            ac = 90.0
            bc = 90.0 - ab
        else:
            bc = np.arccos((b**2 + c**2 - a**2)/(2*b*c)) / np.pi * 180
            ac = np.arccos((a**2 + c**2 - b**2)/(2*a*c)) / np.pi * 180
            
        self.c, self.bc, self.ac = c, bc, ac
        self.c_item.display_value(str(round(self.c, 2)))
        self.bc_item.display_value(str(round(self.bc, 2)))
        self.ac_item.display_value(str(round(self.ac, 2)))
        
    