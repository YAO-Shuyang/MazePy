from PyQt6.QtCore import QUrl, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QSlider
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget


class WheelStepsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Set Wheel Steps")
        self.setGeometry(100, 100, 300, 150)

        self.label = QLabel("Enter the number of steps for the mouse wheel:")
        self.line_edit = QLineEdit()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.line_edit)
        layout.addWidget(self.ok_button)

        self.setLayout(layout)

    def get_wheel_steps(self):
        return int(self.line_edit.text()) if self.result() == QDialog.DialogCode.Accepted else None

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Player")
        self.setGeometry(100, 100, 800, 600)

        # Create the video widget
        self.video_widget = QVideoWidget()

        # Create the media player
        self.media_player = QMediaPlayer()
        self.media_player.setVideoOutput(self.video_widget)

        # Create a slider to control the frame position
        self.frame_slider = QSlider()
        self.frame_slider.setOrientation(Qt.Orientation.Horizontal)
        self.frame_slider.sliderMoved.connect(self.on_slider_moved)

        # Create a button to load the video
        self.load_button = QPushButton("Load Video")
        self.load_button.clicked.connect(self.load_video)

        # Create a button to replay the video
        self.replay_button = QPushButton("Replay")
        self.replay_button.clicked.connect(self.replay_video)

        # Create a button to set mouse wheel steps
        self.steps_button = QPushButton("Set Wheel Steps")
        self.steps_button.clicked.connect(self.set_wheel_steps)

        layout = QVBoxLayout()
        layout.addWidget(self.video_widget)
        layout.addWidget(self.frame_slider)
        layout.addWidget(self.load_button)
        layout.addWidget(self.replay_button)
        layout.addWidget(self.steps_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.wheel_steps = 1

    def load_video(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select a video", "", "Video Files (*.mp4 *.mkv *.avi)")
        if file_path:
            video_url = QUrl.fromLocalFile(file_path)
            self.media_player.setSource(video_url)
            self.media_player.pause()  # Stop playback

            duration = self.media_player.duration()  # Get the duration of the video in milliseconds
            self.frame_slider.setRange(0, duration)  # Set the range of the slider

    def on_slider_moved(self, position):
        duration = self.media_player.duration()
        if duration > 0:
            slider_position = int(position * duration / self.frame_slider.maximum())
            self.media_player.setPosition(slider_position)

    def replay_video(self):
        self.media_player.setPosition(0)  # Reset the video position to the beginning
        self.media_player.pause()  # Start playing the 
        self.frame_slider.setSliderPosition(0)

    def on_wheel_event(self, event):
        delta = event.angleDelta().y()  # Get the wheel movement delta
        slider_value = self.frame_slider.value()
        if delta > 0:
            slider_value -= self.wheel_steps  # Move slider backward
        else:
            slider_value += self.wheel_steps  # Move slider forward

        self.frame_slider.setValue(slider_value)

    def set_wheel_steps(self):
        dialog = WheelStepsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            wheel_steps = dialog.get_wheel_steps()
            if wheel_steps is not None:
                self.wheel_steps = wheel_steps

app = QApplication([])
player = VideoPlayer()
player.show()
app.exec()
