from PyQt6.QtCore import QUrl, Qt, QEvent, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QFileDialog, QSlider, QSpinBox, QProgressBar
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QHBoxLayout
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtGui import QWheelEvent, QMouseEvent
from mazepy.gui.ErrorWindow import ErrorWindow, NoticeWindow
from mazepy.gui.label_video import VideoSliderCoordinator, WheelStepSettor, RecordTable
import numpy as np
import pickle
import pandas as pd
import os
import copy as cp

class VisualizationPlaceFields(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Visualize Place Fields")
        self.setGeometry(100, 100, 1400, 600)
        
        self.file_path, self.folder_dir = None, None

        # Create a button to load the trace.pkl
        self.load_labels = QLabel("Step 1: load trace.pkl")
        self.load_button = QPushButton("Load PKL File")
        self.load_button.clicked.connect(self.load_file)

        # Create a button to select the cell you wanted to plot
        self.select_cell_label = QLabel("Cell ID")
        self.select_cell = QSpinBox()
        
        self.select_cell.valueChanged.connect(self.set_cell_id)

        # Create a horizontal layout for the load widgets
        load_layout = QVBoxLayout()
        load_layout.addWidget(self.load_labels)
        load_layout.addWidget(self.load_button)
        load_layout.addWidget(self.replay_button)
        
    def load_file(self):
        file_dialog = QFileDialog()
        self.file_path, _ = file_dialog.getOpenFileName(self, "Select a pkl file", "", "PKL Files (*.pkl)")
        if self.file_path:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'rb') as handle:
                    trace = pickle.load(handle)
                    
                self.smooth_map_all = cp.deepcopy(trace['smooth_map_all'])
                self.n_neuron = self.smooth_map_all.shape[0]
                print(self.n_neuron)
                self.select_cell.setRange(1, self.n_neuron)
            else:
                ErrorWindow.throw_content(f"{self.file_path} is not existed! Load again!")
        else:
            ErrorWindow.throw_content(f"{self.file_path} is not existed! Load again!")
            
    def set_cell_id(self):
        self.cell_id = self.select_cell.value()
        

if __name__ == '__main__':
    app = QApplication([])
    player = VideoFrameLabel()
    player.show()
    app.exec()