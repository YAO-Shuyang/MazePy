import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QWidget
from mazepy.os.utils import load_yaml

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
        
        # Resize the columns to fit the contents
        self.table_widget.resizeColumnsToContents()

# Example usage
app = QApplication([])
excel_file = r"E:\Data\Cross_maze\cross_maze_paradigm.xlsx"

import pickle, yaml
config = load_yaml('E:\Data\cross_maze_config\config.yaml')
    
headers = config['work sheet header']  # Specify the headers/columns to display
widget = ExcelTableWidget(excel_file, headers, sheet_name='behavior')
widget.show()
app.exec()
