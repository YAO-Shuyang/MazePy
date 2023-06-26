from PyQt6 import QtCore
from PyQt6.QtWidgets import QApplication, QComboBox, QCheckBox, QPushButton, QFileDialog, QHBoxLayout, QScrollArea
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QLineEdit, QWidget
from mazepy.gui import WarningWindow, NoticeWindow
from mazepy.gui import NameList, ConfigFolder, WorkSheet
from mazepy.gui.env_design.designer import EnvDesigner
import yaml
import sys
import pickle
import os

class ConfigCreator(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        
        self.setWindowTitle("Create a configuration file.")
        
        self.config = {}
        
        # 1. Behavior paradigm
        self.behavior_paradigm_tit = QLabel("Enter the name of your behavior paradigm:")
        self.behavior_paradigm_line = QLineEdit()
        self.behavior_paradigm_line.textChanged.connect(self.write_paradigm)
        
        # 2. Experimentor list
        self.experimentor = NameList()
        
        # 3. configuration folder
        self.config_folder = ConfigFolder()
        
        # 4. Select a excel sheet which containing basic information about your experiment sessions/trials.
        self.work_sheet = WorkSheet()
        self.work_sheet.config_dir = self.config_folder.config_dir
        self.config_folder.config_dir_line.returnPressed.connect(self.pass_config_dir_to_sheet)
        self.config_folder.config_dir_button.clicked.connect(self.pass_config_dir_to_sheet)
        
        # 5. Create button
        create_config = QPushButton("Create Configuration File")
        create_config.clicked.connect(self.create_file)
        
        # 6. Go to design environment
        go_to_env_disigner = QPushButton("Go To Design Environment")
        go_to_env_disigner.clicked.connect(self.open_env_designer)        
        
        column_layout = QVBoxLayout()
        column_layout.addWidget(self.behavior_paradigm_tit)
        column_layout.addWidget(self.behavior_paradigm_line)
        column_layout.addLayout(self.experimentor)
        column_layout.addLayout(self.config_folder)
        column_layout.addLayout(self.work_sheet)
        column_layout.addWidget(create_config)
        column_layout.addWidget(go_to_env_disigner)

        central_widget = QWidget()
        central_widget.setLayout(column_layout)
        self.setCentralWidget(central_widget)
        
    def write_paradigm(self):
        """Write the behavior paradigm to the config.
        """
        if self.behavior_paradigm_line.text().isdigit():
            WarningWindow.throw_content("Detect digital: Are you sure to select a digital as the name of the behavior paradigm?")
        
        self.behavior_paradigm = str(self.behavior_paradigm_line.text())
        
    def pass_config_dir_to_sheet(self):
        self.work_sheet.config_dir = self.config_folder.config_dir
        
    def create_file(self):
        if self.work_sheet.check_point_loading_sheet and self.work_sheet.check_point_select_header:
            self.config['behavior paradigm'] = self.behavior_paradigm
            self.config['first name'] = self.experimentor.first_names
            self.config['last name'] = self.experimentor.last_names
            self.config['configuration directory'] = self.config_folder.config_dir
            self.config['work sheet directory'] = self.work_sheet.excel_dir
            self.config['work sheet header'] = self.work_sheet.selected_headers
            
            with open(os.path.join(self.config_folder.config_dir, 'config.pkl'), 'wb') as f:
                pickle.dump(self.config, f)        
                   
            with open(os.path.join(self.config_folder.config_dir, 'config.yaml'), 'w') as f:
                yaml.dump(self.config, f)
            
            NoticeWindow.throw_content("Create configuration file sucessfully! See more details in "+
                                       os.path.join(self.config_folder.config_dir, 'config.yaml'))
        else:
            WarningWindow.throw_content("You have not yet loaded the excel working sheet! Cannot create config file!")
    
    def open_env_designer(self):
        self.designer = EnvDesigner()
        self.designer.show()
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = ConfigCreator()
    editor.show()
    sys.exit(app.exec())
    