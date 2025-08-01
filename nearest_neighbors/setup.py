from distutils.core import setup
#from distutils.extension import Extension
#from Cython.Distutils import build_ext
from setuptools import setup, Extension
from Cython.Build import cythonize


import numpy as np


'''
ext_modules = [Extension(
       "nearest_neighbors",
       sources=["knn.pyx", "knn_.cxx",],  # source file(s)
       include_dirs=["./", numpy.get_include()],
       language="c++",            
       extra_compile_args = [ "-std=c++11", "-fopenmp",],
       extra_link_args=["-std=c++11", '-fopenmp'],
  )]

setup(
    name = "KNN NanoFLANN",
    ext_modules = ext_modules,
    cmdclass = {'build_ext': build_ext},
)
'''
ext_modules = [
    Extension(
        "nearest_neighbors",
        sources=["knn.pyx", "knn_.cxx"],
        include_dirs=[np.get_include(), "."],
        language="c++",
        extra_compile_args=["-std=c++14", "-O3"]
    )
]

setup(
    name="KNN_NanoFLANN",
    ext_modules=cythonize(ext_modules),
)

