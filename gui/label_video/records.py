
from PyQt6.QtCore import QUrl, Qt, QEvent
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QFileDialog, QSlider, QSpinBox
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QHBoxLayout
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtGui import QWheelEvent, QMouseEvent
from mazepy.gui.ErrorWindow import ErrorWindow
from mazepy.gui.label_video import VIDEO_FRAME_DEFAULT, MOUSE_WHEEL_STEP_DEFAULT


class BehaviorEventsRecorder(QVBoxLayout):
    def __ini__(self):
        super().__init__()
        self.save_folder_label = QLabel("Select a folder to save results")
        self.save_folder_button = QPushButton("Select Folder")
        self.save_folder_button.clicked.connect(self.select_save_folder)
        
        scroll_area = QScrollArea()
        
        
    def select_save_folder(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select a video", "", "Video Files (*.mp4 *.mkv *.avi)")
        if file_path:
            self.save_file_directory = file_path
        else:
            ErrorWindow.throw_content("Failed to select a folder! Select again!")
