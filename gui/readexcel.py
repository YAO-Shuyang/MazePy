import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QWidget

class ExcelTableWidget(QWidget):
    def __init__(self, excel_dir: str, headers: list[str], sheet_name: str):
        super().__init__()
        
        self.excel_dir = excel_dir
        
        self.df = pd.read_excel(excel_file)
        
        # Filter the DataFrame to include only the desired columns based on headers
        self.filtered_df = self.df[headers]
        
        # Create a table widget
        self.table_widget = QTableWidget()
        self.table_widget.setRowCount(len(self.filtered_df))
        self.table_widget.setColumnCount(len(headers))
        self.table_widget.setHorizontalHeaderLabels(headers)
        
        # Populate the table widget with data
        for row in range(len(self.filtered_df)):
            for col in range(len(headers)):
                item = QTableWidgetItem(str(self.filtered_df.iat[row, col]))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # Make cells read-only
                self.table_widget.setItem(row, col, item)
        
        # Set the table widget as the central widget of the main window
        self.setCentralWidget(self.table_widget)
        
        # Resize the columns to fit the contents
        self.table_widget.resizeColumnsToContents()

# Example usage
app = QApplication([])
excel_file = r"E:\Data\Cross_maze\cross_maze_paradigm.xlsx"

import pickle, yaml
with open (r"E:\Data\cross_maze_config\config.yaml", 'r') as handle:
    config = yaml.safe_load(handle)
    
headers = config['work sheet header']  # Specify the headers/columns to display
widget = ExcelTableWidget(excel_file, headers)
widget.show()
app.exec()
