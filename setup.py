from setuptools import setup, find_packages
import numpy
setup(
    name="mazepy",
    version="0.1",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    include_dirs=[numpy.get_include()]
)
