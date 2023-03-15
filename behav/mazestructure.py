'''
Some environments have relative complex internal structure, while others are not.

We provide some tools for users to define the internal structure. It contains a
  GUI to help you manually select where to place your internal structure (e.g., 
  Walls or objects)
'''

import pyglet
from mazepy.behav.mazeobj.windows import MainWindow

if __name__ == '__main__':
    main = MainWindow(12, 12)
    pyglet.clock.schedule_interval(main.update, 1 / 60)
    pyglet.app.run()
