from PyQt6.QtWidgets import QVBoxLayout, QSpinBox
from PyQt6.QtWidgets import QVBoxLayout, QLabel, QLineEdit, QPushButton, QHBoxLayout
from PyQt6.QtGui import QWheelEvent
from mazepy.gui.label_video import VIDEO_FRAME_DEFAULT, MOUSE_WHEEL_STEP_DEFAULT

class WheelStepsSettor(QVBoxLayout):
    def __init__(self):
        super().__init__()
      
        # Create a button to set the value of the wheel steps.
        self.current_frame_label = QLabel("Video current frame:")
        spin_box = QSpinBox()
        spin_box.valueChanged.connect(lambda value: print(f"Selected: {value}"))
        self.current_frame_line = QLineEdit()
        self.current_frame_line.returnPressed.connect(self.input_frames)
        self.current_frame_line.wheelEvent = self.on_frame_wheel_event
        
        self.steps_label = QLabel("Mouse Wheel Step Length / frame(s)")
        self.steps_line = QLineEdit()
        self.steps_line.returnPressed.connect(self.input_wheel_steps)
        self.steps_line.wheelEvent = self.on_steplength_wheel_event
        self.steps_set = QPushButton("Set Steps")
        self.steps_set.clicked.connect(self.input_wheel_steps)
        
        self.add_step_button = QPushButton("↑")
        self.add_step_button.clicked.connect(self.add_wheel_steps)
        
        self.sub_step_button = QPushButton("↓")
        self.sub_step_button.clicked.connect(self.sub_wheel_steps)
        
        self.current_frame = VIDEO_FRAME_DEFAULT
        self.current_frame_line.setText(str(VIDEO_FRAME_DEFAULT))
        self.wheel_steps = MOUSE_WHEEL_STEP_DEFAULT
        self.steps_line.setText(str(MOUSE_WHEEL_STEP_DEFAULT))
        
        # Layout these wedgets.
        adjust_buttons_layout = QHBoxLayout()
        adjust_buttons_layout.addWidget(self.add_step_button)
        adjust_buttons_layout.addWidget(self.sub_step_button)
        
        self.addWidget(self.current_frame_label)
        self.addWidget(self.current_frame_line)
        self.addWidget(self.steps_label)
        self.addWidget(self.steps_line)
        self.addLayout(adjust_buttons_layout)
        self.addWidget(self.steps_set)

    def input_wheel_steps(self):
        self.wheel_steps = int(self.steps_line.text()) if self.steps_line.text().isdigit() and int(self.steps_line.text()) >= 0 else MOUSE_WHEEL_STEP_DEFAULT
        self.steps_line.setText(str(self.wheel_steps))
        
    def input_frames(self):
        self.current_frame = int(self.current_frame_line.text()) if self.current_frame_line.text().isdigit() and int(self.current_frame_line.text()) >= 0 else VIDEO_FRAME_DEFAULT
        self.current_frame_line.setText(str(self.current_frame))
        
    def set_frame(self, frame_position: int):
        self.current_frame = frame_position if frame_position >= 0 else 0
        self.current_frame_line.setText(str(frame_position))
        
    def add_wheel_steps(self):
        self.wheel_steps += 1 
        self.steps_line.setText(str(self.wheel_steps))
    
    def sub_wheel_steps(self):
        self.wheel_steps -= 1 if self.wheel_steps > 0 else 0
        self.steps_line.setText(str(self.wheel_steps))
        
    def add_frame_steps(self):
        self.current_frame += 1
        self.current_frame_line.setText(str(self.current_frame))
    
    def sub_frame_steps(self):
        self.current_frame -= 1 if self.current_frame > 0 else 0
        self.current_frame_line.setText(str(self.current_frame))
    
    def on_steplength_wheel_event(self, event: QWheelEvent):
        delta = event.angleDelta().y()
           
        if delta > 0:
            self.add_wheel_steps()
        elif delta < 0:
            self.sub_wheel_steps()
        
    def on_frame_wheel_event(self, event: QWheelEvent):
        delta = event.angleDelta().y()
           
        if delta > 0:
            self.add_frame_steps()
        elif delta < 0:
            self.sub_frame_steps()