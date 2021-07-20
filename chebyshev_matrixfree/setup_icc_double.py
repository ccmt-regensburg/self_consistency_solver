from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

setup(
  name = 'mychebyshev',
  ext_modules=[Extension("cy_chebyshev_cython_double", sources=["cy_chebyshev_cython_double.pyx","chebyshev_final_icc_double.c"], extra_compile_args=['-O2']+['-std=c11']+['-qopenmp']+['-xCORE-AVX2'], extra_link_args=['-qopenmp'], language='c', include_dirs=[numpy.get_include()])],
  cmdclass = {'build_ext': build_ext}
)



