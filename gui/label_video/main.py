from PyQt6.QtCore import QUrl, Qt, QEvent
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QFileDialog, QSlider, QSpinBox
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QHBoxLayout
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtGui import QWheelEvent, QMouseEvent
from mazepy.gui.ErrorWindow import ErrorWindow
from mazepy.gui.label_video import WheelStepsSettor, VideoSliderCoordinator

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
        
if __name__ == '__main__':
    app = QApplication([])
    player = VideoFrameLabel()
    player.show()
    app.exec()

        
