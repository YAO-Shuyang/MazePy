from PyQt6.QtCore import QUrl, Qt, QEvent, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QFileDialog, QSlider, QSpinBox, QProgressBar
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QHBoxLayout
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtGui import QWheelEvent, QMouseEvent
from mazepy.gui.ErrorWindow import ErrorWindow, NoticeWindow
from mazepy.gui.label_video import VideoSliderCoordinator, WheelStepSettor
import cv2
import os
import numpy as np

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

        self.step_settor = WheelStepSettor()
        self.step_settor.wheel_steps_spinbox.valueChanged.connect(self.pass_steps_to_slider)
        self.step_settor.current_stamp_spinbox.valueChanged.connect(self.pass_stamp_to_slider)
        
        self.video_slider.media_player.positionChanged.connect(self.pass_stamp_to_settor)
        
        spacer = QWidget()
        spacer.setFixedSize(10, 10)

        self.load_folder_button = QPushButton("Load Folder")
        self.load_folder_button.clicked.connect(self.load_folder)
        add_frame_button = QPushButton("Add Frame")
        
        add_layout = QHBoxLayout()
        add_layout.addWidget(self.load_folder_button)
        add_layout.addWidget(add_frame_button)
        
        self.progress_bar_label = QLabel("Concatenate videos...")
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        spacer2 = QWidget()
        spacer2.setFixedSize(10, 10)
        
        control_layout = QVBoxLayout()
        control_layout.setContentsMargins(0, 0, 0, 400)
        control_layout.addLayout(load_layout)
        control_layout.addWidget(spacer)
        control_layout.addLayout(self.step_settor)
        control_layout.addWidget(spacer2)
        control_layout.addLayout(add_layout)
        control_layout.addWidget(self.progress_bar_label)
        control_layout.addWidget(self.progress_bar)
                     
        # Create a horizontal layout to hold the control layout and video player layout
        main_layout = QHBoxLayout()
        main_layout.addLayout(control_layout, 1)
        main_layout.addLayout(self.video_slider, 4)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def load_video(self):
        file_dialog = QFileDialog()
        self.file_path, _ = file_dialog.getOpenFileName(self, "Select a video", "", "Video Files (*.mp4 *.mkv *.avi)")
        if self.file_path:
            video_url = QUrl.fromLocalFile(self.file_path)
            self.video_slider.media_player.setSource(video_url)
            self.video_slider.media_player.pause()  # Stop playback

            duration = self.video_slider.media_player.duration()  # Get the duration of the video in milliseconds
            self.video_slider.stamp_slider.setRange(0, duration)  # Set the range of the slider
            
            self.get_frame_info()
            
            self.step_settor.reset_spinbox(duration)
            self.step_settor.wheel_steps_spinbox.valueChanged.connect(self.pass_steps_to_slider)
            self.step_settor.current_stamp_spinbox.valueChanged.connect(self.pass_stamp_to_slider)
            
    def load_folder(self):
        file_dialog = QFileDialog()
        folder_dir = file_dialog.getExistingDirectory(self, "Select a folder", "")
        if folder_dir:
            self.folder_dir = folder_dir
            self.video_files = self.get_video_files()
            self.concat_videos(self.video_files)
        else:
            ErrorWindow.throw_content("Cannot select the folder! Select again.")
    
    def get_video_files(self):
        video_extensions = (".avi")
        video_files = []
        for root, _, files in os.walk(self.folder_dir):
            for file in files:
                if file.lower().endswith(video_extensions):
                    file_path = os.path.join(os.path.abspath(root), file)
                    creation_time = os.path.getctime(file_path)
                    video_files.append((file_path, creation_time))
        video_files.sort(key=lambda x: x[1])  # Sort files based on creation time
        return [file_path for file_path, _ in video_files]

    def update_progress(self, progress):
        self.progress_bar.setValue(progress)
    
    def concat_videos(self, video_files: list[str] = []):
        if len(video_files) == 0:
            ErrorWindow.throw_content(f"This folder ({self.folder_dir}) does not contain any video!")
            return

        # Concatenate videos
        root = os.path.dirname(os.path.abspath(video_files[0]))
        concat_video_path = os.path.join(root, "concatenated_video.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # You can change the codec (e.g., "XVID", "AVC1", "VP90", etc.)
        
        video_num = len(video_files)

        video_writer = None
        for i, video_path in enumerate(self.video_files):
            progress = int((i+1)/video_num*100)
            self.update_progress(progress)
            video_capture = cv2.VideoCapture(video_path)
            fps = int(video_capture.get(cv2.CAP_PROP_FPS))
            frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if video_writer is None:
                # Create a new video file with the first video's properties
                video_writer = cv2.VideoWriter(concat_video_path, fourcc, fps, (frame_width, frame_height))

            while video_capture.isOpened():
                ret, frame = video_capture.read()
                if not ret:
                    break

                # Write each frame to the output video
                video_writer.write(frame)
            
            video_capture.release()

        if video_writer is not None:
            video_writer.release()

        self.file_path = concat_video_path
        video_url = QUrl.fromLocalFile(concat_video_path)
        self.video_slider.media_player.setSource(video_url)
        self.video_slider.media_player.pause()  # Stop playback

        duration = self.video_slider.media_player.duration()  # Get the duration of the video in milliseconds
        self.video_slider.stamp_slider.setRange(0, duration)  # Set the range of the slider
            
        self.get_frame_info()
            
        self.step_settor.reset_spinbox(duration)
        self.step_settor.wheel_steps_spinbox.valueChanged.connect(self.pass_steps_to_slider)
        self.step_settor.current_stamp_spinbox.valueChanged.connect(self.pass_stamp_to_slider)
        
    def get_frame_info(self):
        cap = cv2.VideoCapture(self.file_path)
        total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_to_stamp = np.zeros(total_frame_num, dtype=np.int64)
        stamp_to_frame = np.zeros(self.video_slider.media_player.duration()+1, dtype=np.int64)
        
        NoticeWindow.throw_content("It will take some time to initiate...")
        
        for i in range(total_frame_num):
            ret, frame = cap.read()
            if ret:
                frame_to_stamp[i] = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                if i != 0:
                    stamp_to_frame[frame_to_stamp[i-1]:frame_to_stamp[i]] = i-1
                
                if i == total_frame_num-1:
                    stamp_to_frame[frame_to_stamp[i]::] = i    
            
        NoticeWindow.throw_content("Done.")
        self.step_settor.stamp_to_frame = stamp_to_frame
        self.step_settor.frame_to_stamp = frame_to_stamp
        self.step_settor.total_frame_num = total_frame_num
            
    def pass_steps_to_slider(self):
        self.video_slider.wheel_steps = self.step_settor.wheel_steps
        
    def pass_stamp_to_slider(self):
        self.video_slider.set_stamps(self.step_settor.current_stamp)
        
    def pass_stamp_to_settor(self):
        self.step_settor.set_stamps(self.video_slider.media_player.position())
        
if __name__ == '__main__':
    app = QApplication([])
    player = VideoFrameLabel()
    player.show()
    app.exec()

        
