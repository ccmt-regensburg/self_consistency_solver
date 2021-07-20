import numpy as np
cimport chebyshev_cython_double
cimport numpy as np

def mychebyshev(onsite, pairing, randvec, randvec_nonzero, Nmoments, hopping, onsite_new, pairing_new):
    cdef np.ndarray[double, ndim=1, mode="c"] c_onsite=np.asarray(onsite,dtype=np.float64, order="C")
    cdef np.ndarray[double, ndim=1, mode="c"] c_pairing=np.asarray(pairing,dtype=np.float64, order="C")
    cdef np.ndarray[double, ndim=1, mode="c"] c_randvec=np.asarray(randvec,dtype=np.float64, order="C")
    cdef np.ndarray[double, ndim=1, mode="c"] c_onsite_new=np.asarray(onsite_new,dtype=np.float64, order="C")
    cdef np.ndarray[double, ndim=1, mode="c"] c_pairing_new=np.asarray(pairing_new,dtype=np.float64, order="C")
    chebyshev_cython_double.mychebyshev(&c_onsite[0], &c_pairing[0], &c_randvec[0], randvec_nonzero, Nmoments, hopping, &c_onsite_new[0], &c_pairing_new[0])
    return
