"""
MazePy
=====

Provides
  1. GUIs to design complex environments.
  2. Interfaces to preprocess neural activity collected by calcium 
  imaging, electrophysiological approaches (not yet supported).
  3. Tools and algorithms to analyze neural activity.

How to use the documentation
----------------------------
Documentation is available in the docstrings provided
with the code.

We recommend exploring the docstrings using
`IPython <https://ipython.org>`_, an advanced Python shell with
TAB-completion and introspection capabilities.  See below for further
instructions.

The docstring examples assume that `mazepy` has been imported as `mp`::

  >>> import mazepy as mp

Code snippets are indicated by three greater-than signs::

  >>> x = 42
  >>> x = x + 1

Use the built-in ``help`` function to view a function's docstring::

  >>> help(mp.preprocess)
  ... # doctest: +SKIP

Available subpackages
---------------------
preprocess
    Preprocess raw data
visualize
    Visualize data
decoder
    Decode data
utils
    Utilities for processing neural activity
gui
    GUIs to design complex environments.
"""