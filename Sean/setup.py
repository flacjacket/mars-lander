from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np


extensions = [
    Extension("*",
              sources=["py_utils/images/access_image.pyx"],
              include_dirs=[np.get_include()]),
    Extension("*",
              sources=["py_utils/ml/nn_cy.pyx"],
              include_dirs=[np.get_include()]),
    Extension("*",
              sources=["py_utils/ml/sigmoid.pyx"],
              include_dirs=[np.get_include()]),
]

setup(
    name='Cython builder',
    ext_modules=cythonize(extensions)
)
