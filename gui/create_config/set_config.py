from PyQt6 import QtCore
from PyQt6.QtWidgets import QApplication, QComboBox, QCheckBox, QPushButton, QFileDialog, QHBoxLayout, QScrollArea
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QLineEdit, QWidget
from mazepy.gui import WarningWindow
from mazepy import mkdir
from mazepy.gui.create_config import NAME_NUMBER_DEFAULT, NameList
import yaml
import sys
import os
import pandas as pd
        

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
        self.name_scroll_window = NameList()
        
        # 3. configuration folder
        self.config_folder_tit = QLabel("Enter or select a directory to save you configuration file:")
        self.config_folder_line = QLineEdit()
        self.config_folder_line.returnPressed.connect(self.set_config_folder)
        self.config_folder = QtCore.QDir.homePath()
        self.config_folder_line.setText(self.config_folder)
        self.config_folder_button = QPushButton("Select")
        self.config_folder_button.clicked.connect(self.select_config_folder)

        config_folder_layout = QHBoxLayout()
        config_folder_layout.addWidget(self.config_folder_line, 4)
        config_folder_layout.addWidget(self.config_folder_button, 1)
        
        # 4. Select a excel sheet which containing basic information about your experiment sessions/trials.
        self.work_sheet_tit = QLabel("Load a excel sheet which contains basic information"+
                                     " about your experiment sessions/trials. So-called "+
                                     "basic information might include the directory of "+
                                     "behavioral videos/Mouse Identity/Experiment date or "+
                                     "any other things you think are important.")
        self.work_sheet_tit.setWordWrap(True)
        self.work_sheet_line = QLineEdit()
        self.work_sheet_line.returnPressed.connect(self.set_work_sheet)
        self.work_sheet = None
        self.work_sheet_button = QPushButton("Select")
        self.work_sheet_button.clicked.connect(self.select_work_sheet)
        work_sheet_layout = QHBoxLayout()
        work_sheet_layout.addWidget(self.work_sheet_line, 4)
        work_sheet_layout.addWidget(self.work_sheet_button, 1)
        
        
        column1_layout = QVBoxLayout()
        column1_layout.addWidget(self.behavior_paradigm_tit)
        column1_layout.addWidget(self.behavior_paradigm)
        column1_layout.addLayout(self.name_scroll_window)
        column1_layout.addWidget(self.config_folder_tit)
        column1_layout.addLayout(config_folder_layout)
        column1_layout.addWidget(self.work_sheet_tit)
        column1_layout.addLayout(work_sheet_layout)
        
        column2_layout = QVBoxLayout()
        column2_layout.addWidget(QLabel("2"))
        
        column3_layout = QVBoxLayout()
        column3_layout.addWidget(QLabel("3"))
        
        main_layout = QHBoxLayout()
        main_layout.addLayout(column1_layout,1)
        main_layout.addLayout(column2_layout,1)
        main_layout.addLayout(column3_layout,1)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
    def write_paradigm(self):
        """Write the behavior paradigm to the config.
        """
        if self.behavior_paradigm.text().isdigit():
            WarningWindow.throw_content("Detect digital: Are you sure to select a digital as the name of the behavior paradigm?")
        
        self.config['Behavior paradigm'] = str(self.behavior_paradigm.text())
        
    def select_config_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            None, "Select Folder", self.config_folder, QFileDialog.Option.ShowDirsOnly
        )
        if folder_path:
            self.config_folder = folder_path
            self.config_folder_line.setText(folder_path)
            self.config['config_folder'] = self.config_folder
        else:
            WarningWindow.throw_content(str(folder_path)+" is not loaded sucessfully, select again.")
            
    def set_config_folder(self):
        folder = str(self.config_folder_line.text())
        if QtCore.QDir.isReadable(folder):
            mkdir(folder)
            self.config_folder = folder
        else:
            WarningWindow.throw_content(str(folder)+" is an invalid directory!")
            self.config_folder_line.setText(self.config_folder)
            
    def select_work_sheet(self):
        folder_path, _ = QFileDialog.getOpenFileName(None, "Select working sheet", "", "Excel Files (*.xlsx)")
        if folder_path:
            self.work_sheet = folder_path
            self.work_sheet_line.setText(folder_path)
            self.config['work_sheet'] = self.work_sheet
        else:
            WarningWindow.throw_content(str(folder_path)+" is not loaded sucessfully, select again.")
            
    def set_work_sheet(self):
        folder = str(self.work_sheet_line.text())
        if QtCore.QDir.isReadable(folder):
            mkdir(folder)
            self.work_sheet = folder
        else:
            WarningWindow.throw_content(str(folder)+" is an invalid directory!")
            self.work_sheet_line.setText(self.work_sheet)
        
        
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = ConfigEditor()
    editor.show()
    sys.exit(app.exec())
    