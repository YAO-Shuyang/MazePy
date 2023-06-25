from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QVBoxLayout, QSlider
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtGui import QWheelEvent, QMouseEvent
from mazepy.gui.label_video import VIDEO_FRAME_DEFAULT, MOUSE_WHEEL_STEP_DEFAULT

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
