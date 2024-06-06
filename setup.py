from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "basic._time_sync",
        ["basic\_time_sync.pyx"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "basic._calc_rate",
        ["basic/_calc_rate.pyx"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "basic._utils",
        ["basic/_utils.pyx"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "basic._csmooth",
        ["basic/_csmooth.pyx"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "basic._calculate",
        ["basic/_calculate.pyx"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "basic._cshuffle",
        ["basic/_cshuffle.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

modules = cythonize(extensions)

setup(
    ext_modules=modules,
    include_dirs=[numpy.get_include()],
    py_modules=['os', 'gui', 'plot', 'basic', 'behav', 'neural', 'datastruc']
)

