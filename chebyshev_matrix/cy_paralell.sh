#!/bin/bash

if [ ! -d ./build ]; then
	mkdir build
fi

LINKCC=icc
CC=icc
LDSHARED="icc --shared"

PYTHON_INCLUDE="/home/stosiek/install/anaconda3/include/python3.8/"
NUMPY_INCLUDE="/home/stosiek/install/anaconda3/lib/python3.8/site-packages/numpy/core/include/"
PYTHON_LIB="/home/stosiek/install/anaconda3/lib/"

cython cy_chebyshev_cython_paralell.pyx

icc -I$PYTHON_INCLUDE -I$NUMPY_INCLUDE -DNDEBUG -O2 -Wall -fPIC  -DMKL_ILP64 -c cy_chebyshev_cython_paralell.c -o ./build/cy_chebyshev_cython_paralell.o

icc -shared -Wl,-rpath=$ORIGIN/../lib/:$PYTHON_LIB \
	-L$PYTHON_LIB -L"./" -L$MKLROOT \
	   -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lmkl_rt -lm -ldl -lmkl_avx2 -lsparse_d_mm_chebyshev_new -lmkl_def ./build/cy_chebyshev_cython_paralell.o -o cy_chebyshev_cython_paralell.so

rm -rf build
