from PyQt6.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QPushButton, QAbstractItemView
import numpy as np

class RecordTable(QWidget):
    def __init__(self):
        super().__init__()    

        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(2)
        self.table_widget.setHorizontalHeaderLabels(["Start Frame", "End Frame"])
        self.table_widget.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_widget.itemChanged.connect(self.update_content)
        self.table_widget.clicked.connect(self.current_item)
        
        self.save_records = QPushButton("Save Records")
                
        self.curr_row, self.curr_col = None, None
        self.content = np.array([], np.int64)
        
        layout = QVBoxLayout()
        layout.addWidget(self.table_widget)
        layout.addWidget(self.save_records)
        self.setLayout(layout)
        
    def add_row_at_col1(self, item: int):
        selected_row = self.table_widget.currentRow()
        if selected_row >= 0:
            self.table_widget.insertRow(selected_row)
        else:
            selected_row = self.table_widget.rowCount()
            self.table_widget.insertRow(selected_row)

        self.table_widget.setItem(selected_row, 0, QTableWidgetItem(str(item)))
        self.table_widget.setItem(selected_row, 1, QTableWidgetItem("nan"))
        
    def add_row_at_col2(self, item: int):
        if self.table_widget.item(self.table_widget.rowCount()-1, 1) is not None:
            if self.table_widget.item(self.table_widget.rowCount()-1, 1).text() == "nan":
                print('empty')
                self.table_widget.item(self.table_widget.rowCount()-1, 1).setText(str(int(item)))
            else:
                print('has already got value')
                selected_row = self.table_widget.currentRow()
                if selected_row >= 0:
                    self.table_widget.insertRow(selected_row)
                else:
                    selected_row = self.table_widget.rowCount()
                    self.table_widget.insertRow(selected_row)
                
                self.table_widget.setItem(selected_row, 0, QTableWidgetItem("nan"))
                self.table_widget.setItem(selected_row, 1, QTableWidgetItem(str(item)))
        else:
            selected_row = self.table_widget.currentRow()
            self.table_widget.setItem(selected_row, 1, QTableWidgetItem(str(item)))
            

    def delete_row(self):
        selected_row = self.table_widget.currentRow()
        if selected_row >= 0:
            self.table_widget.removeRow(selected_row)
        else:
            selected_row = self.table_widget.rowCount()
            self.table_widget.removeRow(selected_row-1)
            
    def update_content(self):
        row, col = self.table_widget.rowCount(), self.table_widget.columnCount()
        self.content = np.zeros((row, col), dtype=np.float64)
        
        for r in range(row):
            for l in range(col):
                item = self.table_widget.item(r, l)
                
                if item is not None:
                    if item.text() != "nan":
                        self.content[r, l] = int(item.text())
                    else:
                        self.content[r, l] = np.nan
                else:
                    self.content[r, l] = np.nan
                
        print(1, self.content)
        
    def current_item(self):
        self.curr_row, self.curr_col = self.table_widget.currentRow(), self.table_widget.currentColumn()
        
    def adjust_content(self, num: int):
        if self.curr_row is not None and self.curr_col is not None:
            self.table_widget.item(self.curr_row, self.curr_col).setText(str(num))
        
                

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Dynamic Table Example")
        self.setGeometry(100, 100, 400, 300)

        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(2)
        self.table_widget.setHorizontalHeaderLabels(["Column 1", "Column 2"])
        self.table_widget.itemChanged.connect(self.update_content)
        self.table_widget.clicked.connect(self.current_item)
        self.table_widget.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        self.add_row_button = QPushButton("Add Row")
        self.add_row_button.clicked.connect(self.add_row)

        self.delete_row_button = QPushButton("Delete Row")
        self.delete_row_button.clicked.connect(self.delete_row)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.table_widget)
        main_layout.addWidget(self.add_row_button)
        main_layout.addWidget(self.delete_row_button)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def add_row(self):
        selected_row = self.table_widget.currentRow()
        if selected_row >= 0:
            self.table_widget.insertRow(selected_row)
        else:
            selected_row = self.table_widget.rowCount()
            self.table_widget.insertRow(selected_row)

        self.table_widget.setItem(selected_row, 0, QTableWidgetItem(f"{selected_row+1}"))
        self.table_widget.setItem(selected_row, 1, QTableWidgetItem(f"{selected_row+1}"))

    def delete_row(self):
        selected_row = self.table_widget.currentRow()
        print(selected_row)
        if selected_row >= 0:
            self.table_widget.removeRow(selected_row)
        else:
            selected_row = self.table_widget.rowCount()
            self.table_widget.removeRow(selected_row-1)
            
    def update_input_content(self):
        item = self.table_widget.item(self.curr_row, self.curr_col)
        if item.text().isdigit():
            self.content[self.curr_row, self.curr_col] = float(item.text())
        else:
            item.setText(str(int(self.content[self.curr_row, self.curr_col]))) if not np.isnan(self.content[self.curr_row, self.curr_col]) else item.setText("")
    
    def update_content(self):
        row, col = self.table_widget.rowCount(), self.table_widget.columnCount()
        self.content = np.zeros((row, col), dtype=np.float64)
        
        for r in range(row):
            for l in range(col):
                item = self.table_widget.item(r, l)
                
                if item is not None:
                    self.content[r, l] = float(item.text())
                else:
                    self.content[r, l] = np.nan
                
        print(self.content)
        
    def current_item(self):
        self.curr_row, self.curr_col = self.table_widget.currentRow(), self.table_widget.currentColumn()
        print(self.curr_row, self.curr_col)
        

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
