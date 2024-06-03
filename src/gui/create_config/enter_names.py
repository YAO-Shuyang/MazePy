from PyQt6.QtWidgets import QPushButton, QHBoxLayout, QScrollArea
from PyQt6.QtWidgets import QVBoxLayout, QWidget, QLabel, QLineEdit, QWidget
from mazepy.gui import WarningWindow
from mazepy.gui.create_config import NAME_NUMBER_DEFAULT

class NameList(QVBoxLayout):
    def __init__(self):
        super().__init__()
        
        members_tit = QLabel("Enter the names of the experimentors:")
        self.add_name_button, self.del_name_button = QPushButton("+"), QPushButton("-")
        self.add_name_button.clicked.connect(self.add_name)
        self.del_name_button.clicked.connect(self.del_name)
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.add_name_button)
        buttons_layout.addWidget(self.del_name_button)
        self.first_name_tit, self.comma, self.last_name_tit = QLabel("first name"), QLabel(", "), QLabel("last name")
        name_title_layout = QHBoxLayout()
        name_title_layout.addWidget(self.last_name_tit, 5)
        name_title_layout.addWidget(self.comma, 1)
        name_title_layout.addWidget(self.first_name_tit, 5)
        
        self.first_names, self.last_names, self.name_items = [], [], []
        self.name_num = NAME_NUMBER_DEFAULT-1
        
        self.scroll_lists = QScrollArea()
        
        self.addWidget(members_tit)
        self.addLayout(buttons_layout)
        self.addLayout(name_title_layout)
        self.addWidget(self.scroll_lists)
        
        self.is_init = True
        self.add_name()        
        
    def add_name(self):
        new_name_item = NameItem()
        self.name_items.append(new_name_item)
        new_name_item.first_name_line.textChanged.connect(self.update_names)
        new_name_item.last_name_line.textChanged.connect(self.update_names)
        self.name_num += 1
        
        self.update_scroll_area()
        self.is_init = False
        
    def del_name(self):
        if self.name_num > 1:
            self.name_items.pop()
            self.name_num -= 1
        
        self.update_scroll_area()
        self.is_init = False
            
    def update_names(self):
        self.first_names = []
        for item in self.name_items:
            self.first_names.append(item.first_name)

        self.last_names = []
        for item in self.name_items:
            self.last_names.append(item.last_name)
            
    def update_scroll_area(self):
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setSpacing(0)

        # Add some labels to the layout as an example content
        for item in self.name_items:
            layout.addWidget(item.item)

        # Create a QScrollArea
        scroll_area = self.scroll_lists
        scroll_area.setStyleSheet("background-color: white;")
        scroll_area.setWidgetResizable(False)  # Allow the content to be resized

        # Set the content widget as the child of the QScrollArea
        scroll_area.setWidget(content_widget)
        
class NameItem(object):
    def __init__(self):
        super().__init__()
        
        self.first_name, self.last_name = 'Leo', 'Messi'
        
        self.comma = QLabel(", ")
        self.first_name_line, self.last_name_line = QLineEdit(), QLineEdit()
        self.first_name_line.textChanged.connect(self.get_first_name)
        self.last_name_line.textChanged.connect(self.get_last_name)
        
        layout = QHBoxLayout()
        layout.addWidget(self.last_name_line, 5)
        layout.addWidget(self.comma, 1)
        layout.addWidget(self.first_name_line, 5)
        self.item = QWidget()
        self.item.setLayout(layout)
    
    def get_first_name(self):
        if self.first_name_line.text().isdigit():
            WarningWindow.throw_content("Detect digital: Are you sure to select a digital as the first name?")
        
        self.first_name = str(self.first_name_line.text())
        
    def get_last_name(self):
        if self.last_name_line.text().isdigit():
            WarningWindow.throw_content("Detect digital: Are you sure to select a digital as the last name?")
        
        self.last_name = str(self.last_name_line.text())