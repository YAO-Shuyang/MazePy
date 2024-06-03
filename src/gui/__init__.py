from mazepy.gui.ErrorWindow import ThrowWindow, ErrorWindow, NoticeWindow, WarningWindow
from mazepy.gui.create_config.enter_names import NameItem, NameList
from mazepy.gui.create_config.work_sheet import ConfigFolder, WorkSheet

ENV_OUTSIDE_SHAPE = ['None', 'Circle', 'Square', 'Rectangle', 'Triangle']
""" '8-arm Maze', '6-arm Maze', 'Y Maze', 'T Maze'"""

ENV_SIDE_LENGTH_DEFAULT = 100.0
ENV_ANGLE_DEGREE_DEFAULT = 60.0
LARGEST_LEMGTJ_DEFAULT = 100000.0
SMALLEST_LEMGTJ_DEFAULT = 0.1