from PyQt6.QtWidgets import QVBoxLayout, QWidget, QLabel, QWidget, QDoubleSpinBox
from mazepy.gui import WarningWindow, NoticeWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mazepy import mkdir, clear_spines
from mazepy.gui import ENV_OUTSIDE_SHAPE, ENV_EDGE_LENGTH_DEFAULT, LARGEST_LEMGTJ_DEFAULT, SMALLEST_LEMGTJ_DEFAULT


class ParameterItem(QWidget):
    def __init__(self, label: str) -> None:
        super().__init__()
        
        # input radius
        self.para_label = QLabel(label)
        self.para_label.setWordWrap(True)
        self.para_value_spin = QDoubleSpinBox()
        self.para_value_spin.setSingleStep(0.1)
        self.para_value_spin.valueChanged.connect(self.set_value)
        self.para_value_spin.setRange(SMALLEST_LEMGTJ_DEFAULT, LARGEST_LEMGTJ_DEFAULT)
        self.para_value = ENV_EDGE_LENGTH_DEFAULT
        self.para_value_spin.setValue(ENV_EDGE_LENGTH_DEFAULT)
        
        layout = QVBoxLayout()
        layout.addWidget(self.para_label)
        layout.addWidget(self.para_value_spin)
        
        self.setLayout(layout)
        
    def set_value(self):
        self.para_value = float(self.para_value_spin.text())
        

class PlotStandardShape(QWidget):
    def __init__(self) -> None:
        super().__init__()
        
        # Create a Figure object
        fig = Figure(figsize=(8,8))
        self.ax = clear_spines(fig.add_axes((0.1, 0.1, 0.8, 0.8)), set_invisible_spines='plot')
        self.ax.spines['left'].set_position('zero')
        self.ax.spines['bottom'].set_position('zero')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Create a Canvas widget that will display the Figure
        self.canvas = FigureCanvas(fig)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        self.setFixedSize(300,300)