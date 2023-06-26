from PyQt6 import QtCore
from PyQt6.QtWidgets import QApplication, QComboBox, QCheckBox, QPushButton, QFileDialog, QHBoxLayout, QScrollArea
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QLineEdit, QWidget
from mazepy.gui import WarningWindow, NoticeWindow
from mazepy.gui.env_design.circle import Circle, ParameterItem, PlotStandardShape
from mazepy.gui.env_design.empty import Empty
from mazepy.gui.env_design.square import Square
from mazepy.gui.env_design.rectangle import Rectangle
from mazepy.gui.env_design.triangle import Triangle
from mazepy import mkdir
from mazepy.gui import ENV_OUTSIDE_SHAPE
import yaml
import pickle
import sys
import os

class EnvDesigner(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        
        self.setWindowTitle("Design your environment")
        
        # 1. Load config file
        load_config_label = QLabel("Load the configuration file.")
        load_config_label.setWordWrap(True)
        self.load_config_line = QLineEdit()
        self.config_dir, self.is_config_loaded = "", False
        self.load_config_line.textChanged.connect(self.keep_locked)
        load_config_button = QPushButton("Load Config")
        load_config_button.clicked.connect(self.load_config)
        
        load_layout = QHBoxLayout()
        load_layout.addWidget(self.load_config_line)
        load_layout.addWidget(load_config_button)
        
        # 2. Select a option for the environment shape        
        env_shape_label = QLabel("Select outside shape of the environment")
        self.env_config = {}
        self.env_shape_combo = QComboBox()
        for item in ENV_OUTSIDE_SHAPE:
            self.env_shape_combo.addItem(item)
            self.env_config[item] = {}
        self.env_shape_combo.currentTextChanged.connect(self.display_parameter_selection)
        
        # 3. set data and write to config
        save_config_button = QPushButton("Save")
        save_config_button.clicked.connect(self.write_to_config)
        
        horizontal_layout = QHBoxLayout()
        horizontal_layout.addWidget(self.env_shape_combo)
        horizontal_layout.addWidget(save_config_button)
        
        self.dynamic_obj = Empty()
        
        self.column_layout = QVBoxLayout()
        self.column_layout.addWidget(load_config_label)
        self.column_layout.addLayout(load_layout)
        self.column_layout.addWidget(env_shape_label)
        self.column_layout.addLayout(horizontal_layout)
        self.column_layout.addWidget(self.dynamic_obj)

        central_widget = QWidget()
        central_widget.setLayout(self.column_layout)
        self.setCentralWidget(central_widget)
        
    def keep_locked(self):
        self.load_config_line.setText(self.config_dir)
    
    def load_config(self):
        folder, _ = QFileDialog.getOpenFileName(None, "Select configuration file", "", "PKL Files (*.pkl)")
        if os.path.exists(folder):
            with open(folder, 'rb') as handle:
                self.config = pickle.load(handle)
                self.config_dir = folder
                self.load_config_line.setText(folder)
                if 'environment configuration' in self.config.keys():
                    self.env_config = self.config['environment configuration']
                else:
                    self.config['environment configuration'] = {}
                self.is_config_loaded = True
        else:
            WarningWindow.throw_content(str(folder)+" is not loaded sucessfully, select again.")
    
    def display_parameter_selection(self):
        self.column_layout.removeWidget(self.dynamic_obj)
                 
        self.current_env_shape = self.env_shape_combo.currentText()
        if self.current_env_shape == 'Circle':
            self.dynamic_obj = Circle()
        elif self.current_env_shape == 'Square':
            self.dynamic_obj = Square()
        elif self.current_env_shape == 'Rectangle':
            self.dynamic_obj = Rectangle()
        elif self.current_env_shape == 'Triangle':
            self.dynamic_obj = Triangle()
        else:
            self.dynamic_obj = Empty()
            
        self.column_layout.addWidget(self.dynamic_obj)
        
    def write_to_config(self):
        if self.is_config_loaded:
            self.env_config[self.current_env_shape].update(self.dynamic_obj.identity)
            self.config['environment configuration'].update(self.env_config)
            with open(os.path.join(self.config['configuration directory'], 'config.yaml'), 'w') as f:
                yaml.dump(self.config, f)
            with open(os.path.join(self.config['configuration directory'], 'config.pkl'), 'wb') as f:
                pickle.dump(self.config, f)
            NoticeWindow.throw_content("Save environment information successfully!")
        else:
            WarningWindow.throw_content("You have not yet loaded a config file!")
            

        
            
                
if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = EnvDesigner()
    editor.show()
    sys.exit(app.exec())