import sys
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class YMaze(QVBoxLayout):
    def __init__(self):
        super().__init__()
        
        