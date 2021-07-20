cdef extern from "chebyshev_cython_double.h":
	void mychebyshev(double* onsite, double* pairing, double* randvec, int randvec_nonzero, int Nmoments, double hopping, double* pairing_new, double* onsite_new)
