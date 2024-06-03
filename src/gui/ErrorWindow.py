from PyQt6.QtWidgets import QMessageBox


class ThrowWindow(QMessageBox):
    def __init__(self, parent = None, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self.setGeometry(450, 300, 300, 100)
    
    def set_content(self, content: str):
        self.setText(content)
        
class ErrorWindow(ThrowWindow):
    def __init__(self, parent = None, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self.setWindowTitle('Error Detected!')
        self.setIcon(QMessageBox.Icon.Critical)

    @staticmethod
    def throw_content(content: str):
        Obj = ErrorWindow()
        Obj.set_content(content)
        Obj.exec()

class NoticeWindow(ThrowWindow):
    def __init__(self, parent = None, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self.setWindowTitle('Note')
        self.setIcon(QMessageBox.Icon.Information)

    @staticmethod
    def throw_content(content: str):
        Obj = NoticeWindow()
        Obj.set_content(content)
        Obj.exec()
        
class WarningWindow(ThrowWindow):
    def __init__(self, parent = None, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self.setWindowTitle('Warning!')
        self.setIcon(QMessageBox.Icon.Warning)
        
    @staticmethod
    def throw_content(content: str):
        Obj = WarningWindow()
        Obj.set_content(content)
        Obj.exec()