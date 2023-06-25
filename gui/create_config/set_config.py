from PyQt6 import QtCore
from PyQt6.QtWidgets import QApplication, QComboBox, QCheckBox, QPushButton, QFileDialog, QHBoxLayout, QScrollArea
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QLineEdit, QWidget
from mazepy.gui import WarningWindow
from mazepy import mkdir
from mazepy.gui.create_config import NAME_NUMBER_DEFAULT, NameList, ConfigFolder, WorkSheet
import yaml
import sys
import os

class ConfigEditor(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        
        self.setWindowTitle("Create a configuration file.")
        self.setGeometry(100, 100, 1000, 600)
        
        self.config = {}
        
        # 1. Behavior paradigm
        self.behavior_paradigm_tit = QLabel("Enter the name of your behavior paradigm:")
        self.behavior_paradigm = QLineEdit()
        self.behavior_paradigm.textChanged.connect(self.write_paradigm)
        
        # 2. Experimentor list
        self.experimentor = NameList()
        
        # 3. configuration folder
        self.config_folder = ConfigFolder()
        
        # 4. Select a excel sheet which containing basic information about your experiment sessions/trials.
        self.work_sheet = WorkSheet()
        self.work_sheet.config_dir = self.config_folder.config_dir
        self.config_folder.config_dir_line.returnPressed.connect(self.pass_config_dir_to_sheet)
        self.config_folder.config_dir_button.clicked.connect(self.pass_config_dir_to_sheet)
        
        column1_layout = QVBoxLayout()
        column1_layout.addWidget(self.behavior_paradigm_tit)
        column1_layout.addWidget(self.behavior_paradigm)
        column1_layout.addLayout(self.experimentor)
        column1_layout.addLayout(self.config_folder)
        column1_layout.addLayout(self.work_sheet)
        
        column2_layout = QVBoxLayout()
        column2_layout.addWidget(QLabel("2"))
        
        main_layout = QHBoxLayout()
        main_layout.addLayout(column1_layout,1)
        main_layout.addLayout(column2_layout,2)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
    def write_paradigm(self):
        """Write the behavior paradigm to the config.
        """
        if self.behavior_paradigm.text().isdigit():
            WarningWindow.throw_content("Detect digital: Are you sure to select a digital as the name of the behavior paradigm?")
        
        self.config['Behavior paradigm'] = str(self.behavior_paradigm.text())
        
    def pass_config_dir_to_sheet(self):
        self.work_sheet.config_dir = self.config_folder.config_dir
        

            

if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = ConfigEditor()
    editor.show()
    sys.exit(app.exec())
    