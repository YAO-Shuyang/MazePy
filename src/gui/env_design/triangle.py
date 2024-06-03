import sys
from PyQt6.QtWidgets import QVBoxLayout, QWidget, QLabel, QHBoxLayout, QComboBox
from mazepy.gui.env_design.element import ParameterItem, PlotStandardShape, LockedItem
from mazepy.gui import ErrorWindow, WarningWindow, NoticeWindow
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
        
        canvas_label = QLabel("Counterclockwise arrangement of a, b, c")
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
                    'left side': c,
                    'right side': b,
                    'left angle': self.ac,
                    'right angle': self.ab,
                    'bottom label': 'a',
                    'left label': 'c',
                    'right label': 'b',
                    'vertex': ['A', 'B', 'C']}
        elif b > a and b >= c:
            return {'bottom side': b,
                    'left side': a,
                    'right side': c,
                    'left angle': self.ab,
                    'right angle': self.bc,
                    'bottom label': 'b',
                    'left label': 'a',
                    'right label': 'c',
                    'vertex': ['B', 'C', 'A']}
        elif c > a and c > b:
            return {'bottom side': c,
                    'left side': b,
                    'right side': a,
                    'left angle': self.bc,
                    'right angle': self.ac,
                    'bottom label': 'c',
                    'left label': 'b',
                    'right label': 'a',
                    'vertex': ['C', 'A', 'B']}
    
    def display_parameters(self):
        self.layouts.removeWidget(self.dynamic_widget)
        
        if self.select_mode.currentText() == 'SSS':
            self.dynamic_widget = SSS()
            self.dynamic_widget.a_item.para_value_spin.valueChanged.connect(self.pass_values)
            self.dynamic_widget.b_item.para_value_spin.valueChanged.connect(self.pass_values)
            self.dynamic_widget.c_item.para_value_spin.valueChanged.connect(self.pass_values)
        elif self.select_mode.currentText() == 'SAS':
            print("Change to SAS")
            self.dynamic_widget = SAS()
            self.dynamic_widget.a_item.para_value_spin.valueChanged.connect(self.pass_values)
            self.dynamic_widget.b_item.para_value_spin.valueChanged.connect(self.pass_values)
            self.dynamic_widget.ab_item.para_value_spin.valueChanged.connect(self.pass_values)
        elif self.select_mode.currentText() == 'ASA':
            self.dynamic_widget = ASA()
            self.dynamic_widget.a_item.para_value_spin.valueChanged.connect(self.pass_values)
            self.dynamic_widget.ac_item.para_value_spin.valueChanged.connect(self.pass_values)
            self.dynamic_widget.ab_item.para_value_spin.valueChanged.connect(self.pass_values)
        elif self.select_mode.currentText() == 'HL':
            self.dynamic_widget = HL()
            self.dynamic_widget.a_item.para_value_spin.valueChanged.connect(self.pass_values)
            self.dynamic_widget.c_item.para_value_spin.valueChanged.connect(self.pass_values)
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
        
        self.ax.text(x_t, y_t, label_info['vertex'][0], ha = 'center', va = 'bottom')
        self.ax.text(x_l, 0, label_info['vertex'][1], ha = 'right', va = 'top')
        self.ax.text(x_r, 0, label_info['vertex'][2], ha = 'left',  va = 'top')
        self.canvas.canvas.draw()
        
        self.identity = {
            'Shape': 'Triangle',
            'mode': str(self.select_mode.currentText()),
            'a': self.a, 'b': self.b, 'c': self.c, 'ab': self.ab, 'bc': self.bc, 'ac': self.ac,
            'vertex': {
                label_info['vertex'][0]: (x_t - x_l, y_t, 'upper center'), 
                label_info['vertex'][1]: (0, 0, 'bottom left'),
                label_info['vertex'][1]: (2*x_r, 0, 'bottom right')
                },
            'width': 2*x_r if x_r >= x_t else x_r + x_t,
            'height': y_t,
        }

    def pass_values(self):
        self.a, self.b, self.c = self.dynamic_widget.a, self.dynamic_widget.b, self.dynamic_widget.c
        self.ab, self.bc, self.ac = self.dynamic_widget.ab, self.dynamic_widget.bc, self.dynamic_widget.ac
        self.visualize()

class SSS(QWidget):
    def __init__(self):
        super().__init__()
        
        self.a, self.b, self.c = ENV_SIDE_LENGTH_DEFAULT, ENV_SIDE_LENGTH_DEFAULT, ENV_SIDE_LENGTH_DEFAULT
        
        input_label = QLabel("SSS Mode: using 3 sides length (a, b, c, unit: cm) to ensure a triangle.")
        input_label.setWordWrap(True)
        self.a_item = ParameterItem("Enter a")
        self.a_item.para_value_spin.setValue(ENV_SIDE_LENGTH_DEFAULT)
        self.b_item = ParameterItem("Enter b")
        self.b_item.para_value_spin.setValue(ENV_SIDE_LENGTH_DEFAULT)
        self.c_item = ParameterItem("Enter c")
        self.c_item.para_value_spin.setValue(ENV_SIDE_LENGTH_DEFAULT)
        
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.a_item, 1)
        input_layout.addWidget(self.b_item, 1)
        input_layout.addWidget(self.c_item, 1)
        
        output_label = QLabel("Related values")
        output_label.setWordWrap(True)

        self.ab_item = LockedItem("Angle ∠ab")
        self.bc_item = LockedItem("Angle ∠bc")
        self.ac_item = LockedItem("Angle ∠ac")
     
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.ab_item, 1)
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
        
        self.c_item.para_value_spin.valueChanged.connect(self.set_c)
        self.c_item.para_value_spin.valueChanged.connect(self.calc_values)
        
        self.setLayout(values_layout)
        self.calc_values()
    
    def is_valid(self, a: float, b: float, c: float):
        if a + b > c and a + c > b and b + c > a:
            return True
        else:
            return False
    
    def set_a(self):
        if self.is_valid(float(self.a_item.para_value_spin.value()), self.b, self.c): 
            self.a = float(self.a_item.para_value_spin.value())
        else:
            ErrorWindow.throw_content(f"Sides {float(self.a_item.para_value_spin.value())}, {self.b}, {self.c} "+
                                      "cannot form a triangle, which should satisfy a + b > c, a + c > b, b + c"+
                                      " > a. Input again!")
            self.a_item.para_value_spin.setValue(self.a)

    def set_b(self):
        if self.is_valid(float(self.b_item.para_value_spin.value()), self.a, self.c): 
            self.b = float(self.b_item.para_value_spin.value())
        else:
            ErrorWindow.throw_content(f"Sides {float(self.b_item.para_value_spin.value())}, {self.a}, {self.c} "+
                                      "cannot form a triangle, which should satisfy a + b > c, a + c > b, b + c"+
                                      " > a. Input again!")
            self.b_item.para_value_spin.setValue(self.b)
        
    def set_c(self):
        if self.is_valid(float(self.c_item.para_value_spin.value()), self.b, self.a): 
            self.c = float(self.c_item.para_value_spin.value())
        else:
            ErrorWindow.throw_content(f"Sides {float(self.c_item.para_value_spin.value())}, {self.b}, {self.a} "+
                                      "cannot form a triangle, which should satisfy a + b > c, a + c > b, b + c"+
                                      " > a. Input again!")
            self.c_item.para_value_spin.setValue(self.c)
    
    def calc_values(self):
        a, b, c = self.a, self.b, self.c
        
        ab = np.arccos((a**2 + b**2 - c**2)/(2*a*b)) / np.pi * 180
        bc = np.arccos((b**2 + c**2 - a**2)/(2*b*c)) / np.pi * 180
        ac = np.arccos((a**2 + c**2 - b**2)/(2*a*c)) / np.pi * 180
        
        self.ab, self.bc, self.ac = ab, bc, ac
        
        self.ab_item.display_value(str(round(self.ab, 2)))
        self.bc_item.display_value(str(round(self.bc, 2)))
        self.ac_item.display_value(str(round(self.ac, 2)))

class SAS(QWidget):
    def __init__(self):
        super().__init__()
        
        self.a, self.b, self.ab = ENV_SIDE_LENGTH_DEFAULT, ENV_SIDE_LENGTH_DEFAULT, ENV_ANGLE_DEGREE_DEFAULT
        
        input_label = QLabel("SAS Mode: using 2 sides length (a, b, unit: cm) and their included angle (∠ab, unit: degree) to ensure a triangle.")
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
        self.a = float(self.a_item.para_value_spin.value())

    def set_b(self):
        self.b = float(self.b_item.para_value_spin.value())
        
    def set_ab(self):
        self.ab = float(self.ab_item.para_value_spin.value())
    
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


class ASA(QWidget):
    def __init__(self):
        super().__init__()
        
        self.a, self.ab, self.ac = ENV_SIDE_LENGTH_DEFAULT, ENV_ANGLE_DEGREE_DEFAULT, ENV_ANGLE_DEGREE_DEFAULT
        
        input_label = QLabel("ASA Mode: using 2 angles (∠ab and ∠ac, unit: degree) and their included side (a, unit: cm) to ensure a triangle.")
        input_label.setWordWrap(True)
        self.a_item = ParameterItem("Enter a")
        self.a_item.para_value_spin.setValue(ENV_SIDE_LENGTH_DEFAULT)
        self.ab_item = ParameterItem("Enter ∠ab")
        self.ab_item.para_value_spin.setRange(0.01, 179.99)
        self.ab_item.para_value_spin.setValue(ENV_ANGLE_DEGREE_DEFAULT)
        self.ac_item = ParameterItem("Enter ∠ac")
        self.ac_item.para_value_spin.setRange(0.01, 179.99)
        self.ac_item.para_value_spin.setValue(ENV_ANGLE_DEGREE_DEFAULT)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.a_item, 1)
        input_layout.addWidget(self.ab_item, 1)
        input_layout.addWidget(self.ac_item, 1)
        
        output_label = QLabel("Related values")
        output_label.setWordWrap(True)
        self.b_item = LockedItem("b")
        self.c_item = LockedItem("c")
        self.bc_item = LockedItem("Angle ∠bc")
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.b_item, 1)
        output_layout.addWidget(self.c_item, 1)
        output_layout.addWidget(self.bc_item, 1)
        
        values_layout = QVBoxLayout()
        values_layout.addWidget(input_label)
        values_layout.addLayout(input_layout)
        values_layout.addWidget(output_label)
        values_layout.addLayout(output_layout)
        
        self.a_item.para_value_spin.valueChanged.connect(self.set_a)
        self.a_item.para_value_spin.valueChanged.connect(self.calc_values)

        self.ab_item.para_value_spin.valueChanged.connect(self.set_ab)
        self.ab_item.para_value_spin.valueChanged.connect(self.calc_values)
        
        self.ac_item.para_value_spin.valueChanged.connect(self.set_ac)
        self.ac_item.para_value_spin.valueChanged.connect(self.calc_values)
        
        self.setLayout(values_layout)
        
        self.calc_values()
    
    def set_a(self):
        self.a = float(self.a_item.para_value_spin.value())

    def set_ab(self):
        if float(self.ab_item.para_value_spin.value()) + self.ac < 180:
            self.ab = float(self.ab_item.para_value_spin.value())
        elif float(self.ab_item.para_value_spin.value()) == 90:
            WarningWindow.throw_content("Please using 'HL', 'SSS' or 'SAS' mode to define right triangle.")
            self.ab_item.para_value_spin.setValue(self.ab)
        else:
            ErrorWindow.throw_content(f"Sides {float(self.ab_item.para_value_spin.value())} and {self.ac} "+
                                      "cannot form a triangle, because the sum of them >= 180°. Input again!")
            self.ab_item.para_value_spin.setValue(self.ab)
        
    def set_ac(self):
        if float(self.ac_item.para_value_spin.value()) + self.ab < 180:
            self.ac = float(self.ac_item.para_value_spin.value())
        elif float(self.ac_item.para_value_spin.value()) == 90:
            WarningWindow.throw_content("Please using 'HL', 'SSS' or 'SAS' mode to define right triangle.")
            self.ac_item.para_value_spin.setValue(self.ab)
        else:
            ErrorWindow.throw_content(f"Sides {float(self.ac_item.para_value_spin.value())} and {self.ab} "+
                                      "cannot form a triangle, because the sum of them >= 180°. Input again!")
            self.ac_item.para_value_spin.setValue(self.ac)
    
    def calc_values(self):
        a, ab, ac = self.a, self.ab, self.ac
        
        c = a / np.sin((ac + ab)/180*np.pi) * np.sin(ab/180*np.pi)
        b = c * np.sin(ac/180*np.pi) / np.sin(ab/180*np.pi)
        bc = 180 - ac - ab
            
        self.b, self.c, self.bc = b, c, bc
        self.b_item.display_value(str(round(self.b, 2)))
        self.c_item.display_value(str(round(self.c, 2)))
        self.bc_item.display_value(str(round(self.bc, 2)))
        
        
class HL(QWidget):
    def __init__(self):
        super().__init__()
        
        self.a, self.b, self.c = 80, 60, ENV_SIDE_LENGTH_DEFAULT
        self.ab = 90
        
        input_label = QLabel("HL Mode: specifically for right triangle. Given the side length of the bevel edge c and one of the right-angle edge a.")
        input_label.setWordWrap(True)
        self.a_item = ParameterItem("Enter a")
        self.a_item.para_value_spin.setValue(80)
        self.c_item = ParameterItem("Enter c")
        self.c_item.para_value_spin.setValue(ENV_SIDE_LENGTH_DEFAULT)
        self.b_item = LockedItem("locked b")
        self.b_item.display_value(str(60.00))

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.a_item, 1)
        input_layout.addWidget(self.c_item, 1)
        input_layout.addWidget(self.b_item, 1)
        
        output_label = QLabel("Related angles")
        output_label.setWordWrap(True)
        self.ab_item = LockedItem("Angle ∠ab")
        self.bc_item = LockedItem("Angle ∠bc")
        self.ac_item = LockedItem("Angle ∠ac")
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.ab_item, 1)
        output_layout.addWidget(self.bc_item, 1)
        output_layout.addWidget(self.ac_item, 1)
        
        values_layout = QVBoxLayout()
        values_layout.addWidget(input_label)
        values_layout.addLayout(input_layout)
        values_layout.addWidget(output_label)
        values_layout.addLayout(output_layout)
        
        self.a_item.para_value_spin.valueChanged.connect(self.set_a)
        self.a_item.para_value_spin.valueChanged.connect(self.calc_values)

        self.c_item.para_value_spin.valueChanged.connect(self.set_c)
        self.c_item.para_value_spin.valueChanged.connect(self.calc_values)
        
        self.ab_item.display_value(str(90.00))
        
        self.setLayout(values_layout)
        
        self.calc_values()
    
    def set_a(self):
        if float(self.a_item.para_value_spin.value()) < self.c:
            self.a = float(self.a_item.para_value_spin.value())
        else:
            ErrorWindow.throw_content(f"Right-angled edge ({float(self.a_item.para_value_spin.value())})"+
                                      f" should be never longer than bevel edge ({self.c})! Input again!")
            self.a_item.para_value_spin.setValue(self.a)
            
    def set_c(self):
        if float(self.c_item.para_value_spin.value()) > self.a:
            self.c = float(self.c_item.para_value_spin.value())
        else:
            ErrorWindow.throw_content(f"Right-angled edge ({self.a}) should be never longer than bevel"+
                                      f" edge ({float(self.c_item.para_value_spin.value())})! Input again!")
            self.c_item.para_value_spin.setValue(self.c)
    
    def calc_values(self):
        a, c = self.a, self.c
        
        b = np.sqrt(c**2 - a**2)
            
        self.b, self.bc, self.ac = b, np.arcsin(a/c) / np.pi * 180, np.arcsin(b/c) / np.pi * 180
        self.b_item.display_value(str(round(self.b, 2)))
        self.ac_item.display_value(str(round(self.ac, 2)))
        self.bc_item.display_value(str(round(self.bc, 2)))