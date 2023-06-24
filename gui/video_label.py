from PyQt6.QtCore import QUrl, Qt, QEvent
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QFileDialog, QSlider
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QHBoxLayout
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtGui import QWheelEvent, QMouseEvent
from mazepy.gui.ErrorWindow import ErrorWindow

MOUSE_WHEEL_STEP_DEFAULT = 100
VIDEO_FRAME_DEFAULT = 0

class WheelStepsSettor(QVBoxLayout):
    def __init__(self):
        super().__init__()
      
        # Create a button to set the value of the wheel steps.
        self.current_frame_label = QLabel("Video current frame:")
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
        self.wheel_steps = int(self.steps_line.text()) if self.steps_line.text().isdigit() else MOUSE_WHEEL_STEP_DEFAULT
        self.steps_line.setText(str(self.wheel_steps))
        
    def input_frames(self):
        self.current_frame = int(self.current_frame_line.text()) if self.current_frame_line.text().isdigit() else MOUSE_WHEEL_STEP_DEFAULT
        self.current_frame_line.setText(str(self.current_frame))
        
    def set_frame(self, frame_position: int):
        self.current_frame = frame_position
        self.current_frame_line.setText(str(frame_position))
        
    def add_wheel_steps(self):
        self.wheel_steps += 1
        self.steps_line.setText(str(self.wheel_steps))
    
    def sub_wheel_steps(self):
        self.wheel_steps -= 1
        self.steps_line.setText(str(self.wheel_steps))
        
    def add_frame_steps(self):
        self.current_frame += 1
        self.current_frame_line.setText(str(self.current_frame))
    
    def sub_frame_steps(self):
        self.current_frame -= 1
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


class VideoSliderCoordinator(QVBoxLayout):
    def __init__(self):
        super().__init__()
    
        # Create the video widget
        self.video_widget = QVideoWidget()
        self.on_video, self.on_slider = False, False
        
        # Create the media player
        self.media_player = QMediaPlayer()
        self.media_player.setVideoOutput(self.video_widget)
        self.video_widget.mousePressEvent = self.on_mouse_press_video
        self.video_widget.wheelEvent = self.on_wheel_event
        
        # Create a slider to control the frame position
        self.frame_slider = QSlider()
        self.frame_slider.setOrientation(Qt.Orientation.Horizontal)
        self.frame_slider.sliderMoved.connect(self.on_slider_moved)
        self.frame_slider.sliderPressed.connect(self.on_slider_moved)
        self.frame_slider.mousePressEvent = self.on_mouse_press_slider
        self.frame_slider.mouseMoveEvent = self.on_mouse_press_slider
        self.frame_slider.wheelEvent = self.on_wheel_event
        
        # Create a vertical layout for the video player and slider
        self.addWidget(self.video_widget)
        self.addWidget(self.frame_slider)
                
        self.wheel_steps = MOUSE_WHEEL_STEP_DEFAULT
        self.slider_active = True
    
    def on_video_moved(self, frame_position: int):
        duration = self.media_player.duration()
        if duration > 0:
            slider_position = int(frame_position * self.frame_slider.maximum() / duration)
            self.frame_slider.setSliderPosition(slider_position)
            self.media_player.setPosition(frame_position)
            
    def on_slider_moved(self, slider_position):
        duration = self.media_player.duration()
        if duration > 0:
            frame_position = int(slider_position / self.frame_slider.maximum() * duration)
            self.frame_slider.setSliderPosition(slider_position)
            self.media_player.setPosition(frame_position)

    def replay_video(self):
        self.media_player.setPosition(VIDEO_FRAME_DEFAULT)  # Reset the video position to the beginning
        self.media_player.pause()  # Start playing the video
        self.frame_slider.setSliderPosition(VIDEO_FRAME_DEFAULT)

    def on_wheel_event(self, event: QWheelEvent):
        if self.on_slider or self.on_video:
            delta = event.angleDelta().y()
            frame_position = self.media_player.position()
            
            if delta > 0:
                frame_position += self.wheel_steps
            elif delta < 0:
                frame_position -= self.wheel_steps
            
            self.on_video_moved(frame_position)
            
    def on_mouse_press_video(self, event: QMouseEvent):
        self.on_video, self.on_slider = True, False
        
    def on_mouse_press_slider(self, event: QMouseEvent):
        self.on_video, self.on_slider = False, True
        x = int(event.pos().x() / self.frame_slider.width() * self.frame_slider.maximum())
        self.on_slider_moved(x)
            
    def set_frames(self, frame_position: int):
        self.media_player.setPosition(frame_position)
        self.on_video_moved(frame_position=frame_position)

    
class BehaviorEventsRecorder(QVBoxLayout):
    def __ini__(self):
        super().__init__()
        self.save_folder_label = QLabel("Select a folder to save results")
        self.save_folder_button = QPushButton("Select Folder")
        self.save_folder_button.clicked.connect(self.select_save_folder)
        
        scroll_area = QScrollArea()
        
        
        
    def select_save_folder(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select a video", "", "Video Files (*.mp4 *.mkv *.avi)")
        if file_path:
            self.save_file_directory = file_path
        else:
            ErrorWindow.throw_content("Failed to select a folder! Select again!")

class VideoFrameLabel(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Player")
        self.setGeometry(100, 100, 1000, 600)

        self.video_slider = VideoSliderCoordinator()

        # Create a button to load the video
        self.load_labels = QLabel("Step 1: load video(s)")
        self.load_button = QPushButton("Load Video")
        self.load_button.clicked.connect(self.load_video)

        # Create a button to replay the video
        self.replay_button = QPushButton("Replay")
        self.replay_button.clicked.connect(self.video_slider.replay_video)

        # Create a horizontal layout for the load widgets
        load_layout = QVBoxLayout()
        load_layout.addWidget(self.load_labels)
        load_layout.addWidget(self.load_button)
        load_layout.addWidget(self.replay_button)

        self.step_settor = WheelStepsSettor()
        self.step_settor.steps_set.clicked.connect(self.pass_steps_to_slider)
        self.step_settor.steps_line.textChanged.connect(self.pass_steps_to_slider)
        self.step_settor.steps_line.returnPressed.connect(self.pass_steps_to_slider)
        self.step_settor.current_frame_line.textChanged.connect(self.pass_frame_to_slider)
        self.step_settor.current_frame_line.returnPressed.connect(self.pass_frame_to_slider)
        
        self.video_slider.media_player.positionChanged.connect(self.pass_frame_to_line)
        
        spacer = QWidget()
        spacer.setFixedSize(10, 10)
        control_layout = QVBoxLayout()
        control_layout.setContentsMargins(0, 0, 0, 400)
        control_layout.addLayout(load_layout)
        control_layout.addWidget(spacer)
        control_layout.addLayout(self.step_settor)

        # Create a horizontal layout to hold the control layout and video player layout
        main_layout = QHBoxLayout()
        main_layout.addLayout(control_layout, 1)
        main_layout.addLayout(self.video_slider, 4)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def load_video(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select a video", "", "Video Files (*.mp4 *.mkv *.avi)")
        if file_path:
            video_url = QUrl.fromLocalFile(file_path)
            self.video_slider.media_player.setSource(video_url)
            self.video_slider.media_player.pause()  # Stop playback

            duration = self.video_slider.media_player.duration()  # Get the duration of the video in milliseconds
            self.video_slider.frame_slider.setRange(0, duration)  # Set the range of the slider
            
    def pass_steps_to_slider(self):
        self.video_slider.wheel_steps = self.step_settor.wheel_steps
        
    def pass_frame_to_slider(self):
        self.video_slider.set_frames(self.step_settor.current_frame)
        
    def pass_frame_to_line(self):
        self.step_settor.set_frame(self.video_slider.media_player.position())
    
        

app = QApplication([])
player = VideoFrameLabel()
player.show()
app.exec()

        
