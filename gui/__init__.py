from mazepy.gui.ErrorWindow import ThrowWindow, ErrorWindow, NoticeWindow, WarningWindow
from mazepy.gui.create_config.enter_names import NameItem, NameList
from mazepy.gui.create_config.work_sheet import ConfigFolder, WorkSheet
from mazepy.gui.env_design.element import ParameterItem

ENV_OUTSIDE_SHAPE = ['None', 'Circle', 'Square']
""" '8-arm Maze', '6-arm Maze', 'Y Maze', 'T Maze'"""

ENV_EDGE_LENGTH_DEFAULT = 100.0
LARGEST_LEMGTJ_DEFAULT = 100000.0
SMALLEST_LEMGTJ_DEFAULT = 0.1