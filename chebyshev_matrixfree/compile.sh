CC=icc LINKCC=icc LDSHARED='icc --shared' python setup_icc_double.py build_ext --inplace
mv cy_chebyshev_cython_double.cpython-38-x86_64-linux-gnu.so cy_chebyshev_cython_double.so 
cp cy_chebyshev_cython_double.so ../Modules/
