import pandas as pd
from PyQt6 import QtCore
from PyQt6.QtWidgets import QApplication, QComboBox, QCheckBox, QPushButton, QFileDialog, QHBoxLayout, QScrollArea
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QLineEdit, QWidget
from mazepy.gui import WarningWindow, NoticeWindow
from mazepy import mkdir
from mazepy.gui.create_config import NAME_NUMBER_DEFAULT, NameList
import yaml
import sys
import os
import numpy as np
import shutil


class ConfigFolder(QVBoxLayout):
    def __init__(self) -> None:
        super().__init__()
        self.config_dir_tit = QLabel("Enter or select a directory to save you configuration file:")
        self.config_dir_tit.setWordWrap(True)
        self.config_dir_line = QLineEdit()
        self.config_dir_line.returnPressed.connect(self.set_config_dir)
        self.config_dir = QtCore.QDir.homePath()
        self.config_dir_line.setText(self.config_dir)
        self.config_dir_button = QPushButton("Select")
        self.config_dir_button.clicked.connect(self.select_config_dir)

        config_dir_layout = QHBoxLayout()
        config_dir_layout.addWidget(self.config_dir_line, 4)
        config_dir_layout.addWidget(self.config_dir_button, 1)
        
        self.addWidget(self.config_dir_tit)
        self.addLayout(config_dir_layout)
        
    def select_config_dir(self):
        folder = QFileDialog.getExistingDirectory(
            None, "Select Folder", self.config_dir, QFileDialog.Option.ShowDirsOnly
        )
        if folder:
            self.config_dir = folder
            self.config_dir_line.setText(folder)
        else:
            WarningWindow.throw_content(str(folder)+" is not loaded sucessfully, select again.")
            
    def set_config_dir(self):
        folder = str(self.config_dir_line.text())
        if QtCore.QDir.isReadable(folder):
            mkdir(folder)
            self.config_dir = folder
        else:
            WarningWindow.throw_content(str(folder)+" is an invalid directory!")
            self.config_dir_line.setText(self.config_dir)
            
            

class WorkSheet(QVBoxLayout):
    def __init__(self) -> None:
        super().__init__()
        
        self.work_sheet_tit = QLabel("Load a excel sheet which contains basic information"+
                                     " about your experiment sessions/trials. So-called "+
                                     "basic information might include the directory of "+
                                     "behavioral videos/Mouse Identity/Experiment date or "+
                                     "any other things you think are important.")
        self.work_sheet_tit.setWordWrap(True)
        self.work_sheet_line = QLineEdit()
        self.work_sheet_line.returnPressed.connect(self.set_excel_directory)
        self.excel_dir, self.config_dir = None, None
        
        self.select_excel_button = QPushButton("Select")
        self.select_excel_button.clicked.connect(self.select_excel_directory)
        
        self.select_names = QComboBox()
        self.select_names.addItem("None")
        self.sheet_name = str(self.select_names.currentText())
        self.select_names.currentTextChanged.connect(self.set_sheet_name)
        
        self.load_excel_button = QPushButton("load")
        self.load_excel_button.clicked.connect(self.load_sheet)
        
        self.finish_select_header = QPushButton("Finish Selecting")
        self.finish_select_header.clicked.connect(self.finish_selecting_header)
        
        self.header_list = QScrollArea()
        self.header_list.setStyleSheet("background-color: white;")
        
        select_layout = QHBoxLayout()
        select_layout.addWidget(self.work_sheet_line, 4)
        select_layout.addWidget(self.select_excel_button, 1)
        
        load_layout = QHBoxLayout()
        load_layout.addWidget(self.select_names, 4)
        load_layout.addWidget(self.load_excel_button, 1)
        
        self.addWidget(self.work_sheet_tit)
        self.addLayout(select_layout)
        self.addLayout(load_layout)
        self.addWidget(self.header_list)
        self.addWidget(self.finish_select_header)
        
    def select_excel_directory(self):
        folder, _ = QFileDialog.getOpenFileName(None, "Select working sheet", "", "Excel Files (*.xlsx)")
        if os.path.exists(folder):
            self.excel_dir = folder
            self.work_sheet_line.setText(folder)
            self.load_sheet_name()
        else:
            WarningWindow.throw_content(str(folder)+" is not loaded sucessfully, select again.")
            
    def set_excel_directory(self):
        folder = str(self.work_sheet_line.text())
        if os.path.exists(folder):
            self.work_sheet = folder
            self.load_sheet_name()
        else:
            WarningWindow.throw_content(str(folder)+" is an invalid directory!")
            self.work_sheet_line.setText(self.work_sheet)
    
    def load_sheet_name(self):
        if os.path.exists(self.excel_dir):
            file = pd.ExcelFile(self.excel_dir)
            self.select_names.clear()
            for n in file.sheet_names:
                self.select_names.addItem(n)
                
    def set_sheet_name(self):
        self.sheet_name = str(self.select_names.currentText())
    
    def load_sheet(self):
        if os.path.exists(self.excel_dir) and self.sheet_name != 'None':
            self.f = pd.read_excel(self.excel_dir, sheet_name=self.sheet_name)
            
            self.headers = self.f.columns
            
            widgets_content = QWidget()
            
            self.headers_to_checkbox = {}
            header_layout = QVBoxLayout(widgets_content)
        
            for i, header in enumerate(self.headers):
                if i % 2 == 0:
                    line_layout = QHBoxLayout()
            
                checker = QCheckBox(str(header))
                checker.clicked.connect(self.select_headers)
            
                self.headers_to_checkbox[header] = [checker, 0]
            
                line_layout.addWidget(checker)
            
                if i % 2 == 1 or i == len(self.headers)-1:
                    header_layout.addLayout(line_layout)

            self.header_list.setWidget(widgets_content)
            self.select_headers()
           
    def select_headers(self):
        self.selected_headers = []
        for i, header in enumerate(self.headers):
            checker = self.headers_to_checkbox[header][0]
            if checker.isChecked():
                self.selected_headers.append(header)
        
    def finish_selecting_header(self):
        self.f = self.f[self.selected_headers]
        self.f.to_excel(os.path.join(self.config_dir, os.path.basename(self.excel_dir)), index=False)
        self.excel_dir = os.path.join(self.config_dir, os.path.basename(self.excel_dir))