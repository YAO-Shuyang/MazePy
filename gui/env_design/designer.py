from PyQt6 import QtCore
from PyQt6.QtWidgets import QApplication, QComboBox, QCheckBox, QPushButton, QFileDialog, QHBoxLayout, QScrollArea
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QLineEdit, QWidget
from mazepy.gui import WarningWindow, NoticeWindow
from mazepy import mkdir
from mazepy.gui import ENV_OUTSIDE_SHAPE
import yaml
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
        self.config_dir, self.is_read_config = "", False
        self.load_config_line.textChanged.connect(self.keep_locked)
        load_config_button = QPushButton("Load Config")
        load_config_button.clicked.connect(self.load_config)
        
        load_layout = QHBoxLayout()
        load_layout.addWidget(self.load_config_line)
        load_layout.addWidget(load_config_button)
        
        # 2. Select a option for the environment shape        
        env_shape_label = QLabel("Select outside shape of the environment")
        self.env_shape_combo = QComboBox()
        for item in ENV_OUTSIDE_SHAPE:
            self.env_shape_combo.addItem(item)           
        self.env_shape_combo.currentTextChanged.connect(self.display_parameter_selection)
        
        # 3. Display dinamic widgets on the gui based on your selection
        self.dynamic_widgets = QWidget()
        
        column_layout = QVBoxLayout()
        column_layout.addWidget(load_config_label)
        column_layout.addLayout(load_layout)
        column_layout.addWidget(env_shape_label)
        column_layout.addWidget(self.env_shape_combo)
        column_layout.addWidget(self.dynamic_widgets)

        central_widget = QWidget()
        central_widget.setLayout(column_layout)
        self.setCentralWidget(central_widget)
        
    def keep_locked(self):
        self.load_config_line.setText(self.config_dir)
    
    def load_config(self):
        folder, _ = QFileDialog.getOpenFileName(None, "Select configuration file", "", "YALM Files (*.yaml)")
        if os.path.exists(folder):
            with open(folder, 'r') as handle:
                self.config = yaml.safe_load(handle)
                self.config_dir = folder
                self.load_config_line.setText(folder)
        else:
            WarningWindow.throw_content(str(folder)+" is not loaded sucessfully, select again.")
    
    def display_parameter_selection(self):
        print(self.env_shape_combo.currentText())
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = EnvDesigner()
    editor.show()
    sys.exit(app.exec())