import typing
from PyQt6 import QtCore
from PyQt6.QtCore import QUrl, Qt, QEvent
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QFileDialog, QSlider
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QHBoxLayout
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtGui import QWheelEvent, QMouseEvent


class ThrowWindow(QDialog):
    def __init__(self, parent = None, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self.setGeometry(450, 300, 300, 100)
    
    def set_content(self, content: str):
        content = QLabel(content)
        layout = QHBoxLayout()
        layout.addWidget(content)
        
class ErrorWindow(ThrowWindow):
    def __init__(self, parent = None, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self.setWindowTitle('Error Detected!')

    @staticmethod
    def throw_content(content: str):
        Obj = ErrorWindow()
        Obj.set_content(content)

class NoticeWindow(ThrowWindow):
    def __init__(self, parent = None, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self.setWindowTitle('Note')

    @staticmethod
    def throw_content(content: str):
        Obj = NoticeWindow()
        Obj.set_content(content)
        
class WarningWindow(ThrowWindow):
    def __init__(self, parent = None, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self.setWindowTitle('Warning!')
        
    @staticmethod
    def throw_content(content: str):
        Obj = WarningWindow()
        Obj.set_content(content)