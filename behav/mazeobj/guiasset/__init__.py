'''
This file contains several pictures and objects to make up the 
  background or basic settings of the GUI.
'''
import pyglet
import os

def load_image(file, *args, **kwargs):
    img = pyglet.resource.image(file, *args, **kwargs)
    img.anchor_x = img.width // 2
    img.anchor_y = img.height // 2
    return img

working_dir = os.path.dirname(os.path.realpath(__file__))
pyglet.resource.path = [working_dir]
pyglet.resource.reindex()
BACKGROUND_IMG = load_image("Background.png")