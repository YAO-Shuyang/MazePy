from PyQt6.QtWidgets import QVBoxLayout, QSpinBox
from PyQt6.QtWidgets import QVBoxLayout, QLabel, QLineEdit, QPushButton, QHBoxLayout
from PyQt6.QtGui import QWheelEvent
from mazepy.gui.label_video import VIDEO_FRAME_DEFAULT, VIDEO_STAMP_DEFAULT, MOUSE_WHEEL_STEP_DEFAULT

class WheelStepSettor(QVBoxLayout):
    def __init__(self):
        super().__init__()
        
        self.current_stamp_spinbox = QSpinBox()
        self.current_frame_spinbox = QSpinBox()
        self.wheel_steps_spinbox = QSpinBox()
        self.stamp_to_frame, self.frame_to_stamp, self.total_frame_num = None, None, 0
        
        # Create spin boxes to set/show time values
        self.current_stamp_label = QLabel("Video current time stamp / ms")
        self.current_stamp = VIDEO_STAMP_DEFAULT
        
        self.current_frame_label = QLabel("Video current frame")
        self.current_frame = VIDEO_FRAME_DEFAULT
        
        self.wheel_steps_label = QLabel("Mouse Wheel Step Length / ms")
        self.wheel_steps = MOUSE_WHEEL_STEP_DEFAULT
        self.reset_spinbox(100000)
        
        self.addWidget(self.current_stamp_label)
        self.addWidget(self.current_stamp_spinbox)
        self.addWidget(self.current_frame_label)
        self.addWidget(self.current_frame_spinbox)
        self.addWidget(self.wheel_steps_label)
        self.addWidget(self.wheel_steps_spinbox)
        
    def set_stamps(self, stamp_position: int):
        self.current_stamp = stamp_position
        self.current_stamp_spinbox.setValue(self.current_stamp)
            
    def input_stamps(self):
        self.current_stamp = self.current_stamp_spinbox.value()
        self.current_stamp_spinbox.setValue(self.current_stamp)
        
    def input_steps(self):
        self.wheel_steps = self.wheel_steps_spinbox.value()
        self.wheel_steps_spinbox.setValue(self.wheel_steps)
        
    def stamp_set_frame(self):
        self.current_frame = self.stamp_to_frame[self.current_stamp_spinbox.value()]
        self.current_frame_spinbox.setValue(self.current_frame)
        
    def frame_set_stamps(self):
        self.current_stamp = self.frame_to_stamp[self.current_frame_spinbox.value()]
        self.current_stamp_spinbox.setValue(self.current_stamp)
        
    def reset_spinbox(self, total_stamps: int):
        self.current_stamp_spinbox.setRange(0, total_stamps)
        self.current_stamp_spinbox.setValue(self.current_stamp)
        self.current_stamp_spinbox.valueChanged.connect(self.input_stamps)
        self.current_stamp_spinbox.valueChanged.connect(self.stamp_set_frame)
        
        self.wheel_steps_spinbox.setRange(0, total_stamps)
        self.wheel_steps_spinbox.setValue(self.wheel_steps)
        self.wheel_steps_spinbox.valueChanged.connect(self.input_steps)

        self.current_frame_spinbox.setRange(0, self.total_frame_num)
        self.current_frame_spinbox.setValue(self.current_frame)
        self.current_frame_spinbox.valueChanged.connect(self.frame_set_stamps)