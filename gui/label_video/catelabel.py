from logging import warn
from PyQt6.QtCore import QUrl, Qt, QEvent, QThread, pyqtSignal, QObject
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QFileDialog, QSlider, QSpinBox, QProgressBar
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QHBoxLayout, QMessageBox, QInputDialog
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtGui import QWheelEvent, QMouseEvent
from mazepy.gui.ErrorWindow import ErrorWindow, NoticeWindow
from mazepy.gui.label_video import VideoSliderCoordinator, WheelStepSettor, RecordTable
import numpy as np
import cv2
import pickle
import pandas as pd
import time
import os

from sympy import content

class ClickFilter(QObject):
    def __init__(self, spinbox):
        super().__init__()
        self.spinbox = spinbox

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseButtonPress:
            # 如果点击的不是 spinbox 且当前 spinbox 有焦点
            if QApplication.focusWidget() == self.spinbox and obj != self.spinbox:
                self.spinbox.clearFocus()
        return super().eventFilter(obj, event)

class VideoFrameCategoryLabel(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Player")
        self.setGeometry(100, 100, 1400, 700)
        
        self.file_path, self.folder_dir = None, None
        self.content = np.zeros((0, 2), dtype=np.int64)
        self.is_loaded = False

        self.video_slider = VideoSliderCoordinator()

        # Create a button to load the video
        self.load_labels = QLabel("Step 1: load video(s)")
        self.load_button = QPushButton("Load Video")
        self.load_button.clicked.connect(self.load_video)
        # If the video is not present, select the folder and concatenate the videos
        self.load_folder_button = QPushButton("Load Folder")
        self.load_folder_button.clicked.connect(self.load_folder)    
        self.load_excel_button = QPushButton("Load Excel")
        self.load_excel_button.clicked.connect(self.load_excel)
        self.progress_bar_label = QLabel("Concatenate videos...")
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)      
        
        # Create a button to replay the video
        self.replay_button = QPushButton("Replay")
        self.replay_button.clicked.connect(self.video_slider.replay_video)

        # Create a horizontal layout for the load widgets
        load_layout = QVBoxLayout()
        load_layout.addWidget(self.load_labels)
        load_layout.addWidget(self.load_button)
        load_layout.addWidget(self.load_excel_button)
        load_layout.addWidget(self.load_folder_button)
        load_layout.addWidget(self.progress_bar_label)
        load_layout.addWidget(self.progress_bar)
        load_layout.addWidget(self.replay_button)
         
        # Define the number of categories
        n_cate_reminer = QLabel("Define the No. of categories here:")
        n_cate_layout = QHBoxLayout()
        self.n_cate = 1
        self.cate_definer = QSpinBox()
        self.cate_definer.setRange(1, 9)
        self.cate_definer_confirmation = QPushButton("Confirm")
        self.cate_definer_confirmation.clicked.connect(self.confirm_n_cate)
        self.cate_definer_resettor = QPushButton("Reset")
        self.cate_definer_resettor.setEnabled(False)
        self.cate_definer_resettor.clicked.connect(self.reset_n_cate)
        n_cate_layout.addWidget(self.cate_definer)
        n_cate_layout.addWidget(self.cate_definer_confirmation)
        n_cate_layout.addWidget(self.cate_definer_resettor)
        n_cate_explanation = QLabel(
            "Click the 'Confirm' button to confirm the number of categories. If not, "
            "further process will not be available."
        )
        n_cate_explanation.setWordWrap(True)

        # Set the steps of the wheel that controls the movement of the slider
        self.step_settor = WheelStepSettor()
        self.step_settor.wheel_steps_spinbox.valueChanged.connect(self.pass_steps_to_slider)
        self.step_settor.current_stamp_spinbox.valueChanged.connect(self.pass_stamp_to_slider)
        self.step_settor.current_stamp_spinbox.lineEdit().setReadOnly(True)
        self.step_settor.current_frame_spinbox.lineEdit().setReadOnly(True)
        
        self.video_slider.media_player.positionChanged.connect(self.pass_stamp_to_settor)
        
        spacer = QWidget()
        spacer.setFixedSize(10, 10)
        
        self.record_sheet = RecordTable(column_labels=["Frame ID", "Category ID"])
        self.record_sheet.save_records.clicked.connect(self.save_labeling_records)
        
        modulate_note = QLabel(
            "Note: Move the video to certain frame and then press Buttons 1 to 9 to select"
            " the category you want to label for that frame. If you want to delete the label, "
            "move the video to that frame and press the 'Delete' button. "
        )
        modulate_note.setWordWrap(True)
        self.delete = QPushButton("Delete")
        self.delete.clicked.connect(self.record_sheet.delete_row)
             
        spacer2 = QWidget()
        spacer2.setFixedSize(10, 10)
        
        control_layout = QVBoxLayout()
        #control_layout.setContentsMargins(0, 0, 0, 100)
        control_layout.addLayout(load_layout)
        control_layout.addWidget(n_cate_reminer)
        control_layout.addLayout(n_cate_layout)
        control_layout.addWidget(n_cate_explanation)
        control_layout.addWidget(spacer)
        control_layout.addLayout(self.step_settor)
        control_layout.addWidget(spacer2)
        control_layout.addWidget(modulate_note)
        control_layout.addWidget(self.delete)
                     
        # Create a horizontal layout to hold the control layout and video player layout
        main_layout = QHBoxLayout()
        main_layout.addLayout(control_layout, 1)
        main_layout.addLayout(self.video_slider, 4)
        main_layout.addWidget(self.record_sheet)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        self.filter_current_stamp = ClickFilter(self.step_settor.current_stamp_spinbox)
        self.filter_current_frame = ClickFilter(self.step_settor.current_frame_spinbox)
        self.filter_wheel_steps = ClickFilter(self.step_settor.wheel_steps_spinbox)
        self.installEventFilter(self.filter_current_stamp)
        self.installEventFilter(self.filter_current_frame)
        self.installEventFilter(self.filter_wheel_steps)
        central_widget.installEventFilter(self.filter_current_stamp)
        central_widget.installEventFilter(self.filter_current_frame)
        central_widget.installEventFilter(self.filter_wheel_steps)

    def confirm_n_cate(self):
        self.n_cate = self.cate_definer.value()
        self.cate_definer_confirmation.setEnabled(False)
        self.cate_definer.setEnabled(False)
        self.cate_definer_resettor.setEnabled(True)
        
    def reset_n_cate(self):
        # Warn user that defining new categories will clear existing labels
        warn = QMessageBox()
        warn.setIcon(QMessageBox.Icon.Warning)
        warn.setWindowTitle("Confirm Category Reset")
        warn.setText("Redefining categories will clear all existing labels. Proceed?")
        warn.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        warn.setDefaultButton(QMessageBox.StandardButton.Cancel)
        result = warn.exec()
        if result == QMessageBox.StandardButton.Ok:
            self.n_cate = 1
            self.content = np.zeros((0, 2), dtype=np.int64)
            self.cate_definer.setValue(self.n_cate)
            self.cate_definer_confirmation.setEnabled(True)
            self.cate_definer.setEnabled(True)
            self.cate_definer_resettor.setEnabled(False)
            self.record_sheet.clear_content()
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), 'Reset Categories.')

    def load_video(self):
        file_dialog = QFileDialog()
        self.file_path, _ = file_dialog.getOpenFileName(self, "Select a video", "", "Video Files (*.mp4 *.mkv *.avi)")
        if self.file_path:
            self.is_loaded = True
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
            self.is_loaded = True
        else:
            ErrorWindow.throw_content("Cannot select the folder! Select again.")
    
    def load_excel(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select an Excel file", "", "Excel Files (*.xlsx)")
        if file_path:
            file = pd.read_excel(file_path)
            self.content = file.to_numpy()[:, :2]
            self.record_sheet.set_content(self.content)
            self.n_cate = int(file['No. of categories'][0])
            self.cate_definer.setValue(self.n_cate)
            self.cate_definer_confirmation.setEnabled(False)
            self.cate_definer.setEnabled(False)
            self.cate_definer_resettor.setEnabled(True)
        else:
            ErrorWindow.throw_content("Cannot select the Excel file! Select again.")
            

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
        
    def pass_frame_to_sheet(self):
        # pass adjusted frame to sheet
        if self.record_sheet.curr_col is not None and self.record_sheet.curr_row is not None:
            self.record_sheet.adjust_content(int(self.step_settor.current_frame))
        else:
            NoticeWindow.throw_content("Please select the bin you want to adjust first!")
            
    def save_labeling_records(self):
        root = os.path.dirname(self.file_path)
        with open(os.path.join(root, 'frame_labels.pkl'), 'wb') as f:
            pickle.dump(self.record_sheet.content, f)
            
        Data = {'Frame ID': self.record_sheet.content[:, 0], 'Category ID': self.record_sheet.content[:, 1]}
        n_cate = np.full_like(Data['Frame ID'], np.nan)
        n_cate[0] = self.n_cate
        Data['No. of categories'] = n_cate
        
        D = pd.DataFrame(Data)
        D.to_excel(os.path.join(root, 'frame_labels.xlsx'), sheet_name='labeled frames', index=False)
            
        NoticeWindow.throw_content(f"Save file successfully as {os.path.join(root, 'frame_labels.pkl')} and {os.path.join(root, 'frame_labels.xlsx')}")
    
    def add_renew_content(self, num: int):
        curr_frame = self.step_settor.current_frame_spinbox.value()
        
        # If current frame has already been labeled, overwrite the label
        if curr_frame in self.content[:, 0]:
            self.content[self.content[:, 0] == curr_frame, 1] = num
        else:
            self.content = np.append(self.content, np.array([[curr_frame, num]]), axis=0)
        
        self.content = self.content[np.argsort(self.content[:, 0]), :]
        self.record_sheet.set_content(self.content)
        
    def delete_renew_content(self):
        curr_frame = self.step_settor.current_frame_spinbox.value()
        
        if curr_frame in self.content[:, 0]:
            # If current frame has already been labeled, delete the label
            self.content = np.delete(self.content, np.where(self.content[:, 0] == curr_frame), axis=0)
            self.record_sheet.set_content(self.content)
    
    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key in [Qt.Key.Key_1, Qt.Key.Key_2, Qt.Key.Key_3, Qt.Key.Key_4, Qt.Key.Key_5, 
                   Qt.Key.Key_6, Qt.Key.Key_7, Qt.Key.Key_8, Qt.Key.Key_9][:self.n_cate] and self.is_loaded:
            # Check if the number of categories has been defined
            if self.cate_definer_confirmation.isEnabled():
                warn = QMessageBox()
                warn.setIcon(QMessageBox.Icon.Warning) 
                warn.setWindowTitle("Warning")
                warn.setText("Please define the number of categories first!")
                warn.setStandardButtons(QMessageBox.StandardButton.Ok)
                warn.exec()
                return
            else:
                cate = key - Qt.Key.Key_0
                self.add_renew_content(cate)
        elif (key == Qt.Key.Key_Delete or key == Qt.Key.Key_Backspace) and self.is_loaded:
            self.delete_renew_content()
        #left and right arrow keys
        elif key in [Qt.Key.Key_Left, Qt.Key.Key_Down] and self.is_loaded:
            self.step_settor.current_frame_spinbox.setValue(self.step_settor.current_frame_spinbox.value() - 1)
        elif key in [Qt.Key.Key_Right, Qt.Key.Key_Up] and self.is_loaded:
            self.step_settor.current_frame_spinbox.setValue(self.step_settor.current_frame_spinbox.value() + 1)
        else:
            super().keyPressEvent(event)
          
    def wheelEvent(self, event: QWheelEvent):
        if self.is_loaded:
            step = self.step_settor.wheel_steps_spinbox.value()
            if event.angleDelta().y() < 0:
                self.step_settor.current_stamp_spinbox.setValue(max(self.step_settor.current_stamp_spinbox.value() - step, 0))
            else:
                self.step_settor.current_stamp_spinbox.setValue(min(
                    self.step_settor.current_stamp_spinbox.value() + step, self.video_slider.media_player.duration()
                ))
        else:
            super().wheelEvent(event)
      
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = VideoFrameCategoryLabel()
    window.show()
    sys.exit(app.exec())
