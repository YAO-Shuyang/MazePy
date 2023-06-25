import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create a Figure object
        fig = Figure(figsize=(8,6))
        fig.add_subfigure()
        ax = plt.axes()
        ax.plot([1,2], [2, 3])
        
        # Create a Canvas widget that will display the Figure
        canvas = FigureCanvas(fig)

        # Create a QVBoxLayout to hold the Canvas widget
        layout = QVBoxLayout()
        layout.addWidget(canvas)

        # Create a central widget and set the layout
        central_widget = QWidget()
        central_widget.setLayout(layout)

        # Set the central widget for the main window
        self.setCentralWidget(central_widget)


app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())