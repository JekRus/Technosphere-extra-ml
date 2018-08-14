import numpy
from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'Tree',
  ext_modules = cythonize("DecisionTree2.pyx"),
  include_dirs=[numpy.get_include()]
)

