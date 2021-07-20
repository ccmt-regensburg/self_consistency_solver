#!python
#cython: language_level=3

import numpy as np
import scipy.sparse as sps
cimport numpy as np
#cimport scipy.sparse as sps
import time

cdef extern from "mkl_service.h":
	cdef void mkl_get_version_string (char* buf, int len) nogil
#cdef extern from "mkl_spblas.h":
#	cdef void mkl_dcsrmv(const char *transa, const long long *m, const long long *k, const double *alpha, const char *matdescra, const double *val, const long long *indx, const long  long *pntrb, const long #long *pntre, const double *x, const double *beta, double *y)

cdef extern from "sparse_d_mm_chebyshev_new.h":
	cdef void sparse_d_mm_chebyshev(double *AVal, long long *AColInd, long long *ARowPtrB, long long  *ARowPtrE, double* x, long long columns, long long ldx, double* y, long long ldy, long long niter, long long index, double* momentspairing, double* momentsonsite,long long sitenumber);




def mychebyshev(data, indices,pntrb,pntre,Nm,processnumber, xx,xxpro,basisvecv,basisvecu,momentsveconsite,momentsvecpairing,length):
	#time1=time.time()
	cdef np.ndarray[double, ndim=1, mode="c"] c_val=np.asarray(data,dtype=np.float64, order="C")
	cdef np.ndarray[long long int, ndim=1, mode="c"] c_indx=np.asarray(indices,dtype=np.int64, order="C") 
	cdef np.ndarray[long long int, ndim=1, mode="c"] c_pntrb=np.asarray(pntrb,dtype=np.int64, order="C")
	cdef np.ndarray[long long int, ndim=1, mode="c"] c_pntre=np.asarray(pntre,dtype=np.int64, order="C")
	cdef long long int c_Nm=Nm
	cdef long long int c_Nmhalf=Nm//2
	cdef long long int c_process=processnumber
	cdef long long int c_xx=xx
	cdef long long int c_xxpro=xxpro
	cdef long long int c_length=length
	cdef long long int c_half=length//2
	cdef long long int c_index=c_xxpro*c_process		
	cdef np.ndarray[double, ndim=1, mode="c"] c_basisvec=np.asarray(basisvecv,dtype=np.float64, order="C")
	cdef np.ndarray[double, ndim=1, mode="c"] c_basisveccopy=np.asarray(basisvecu,dtype=np.float64, order="C")
	cdef np.ndarray[double, ndim=1, mode="c"] c_testcopy=np.zeros(xx*length,dtype=np.float64, order="C")
	cdef np.ndarray[double, ndim=1, mode="c"] c_onsite_new=np.asarray(momentsveconsite,dtype=np.float64, order="C")
	cdef np.ndarray[double, ndim=1, mode="c"] c_pairing_new=np.asarray(momentsvecpairing,dtype=np.float64, order="C")
	#print(type(c_val[0]))
	#print(c_xx)
	sparse_d_mm_chebyshev(&c_val[0], &c_indx[0], &c_pntrb[0], &c_pntre[0], &c_basisvec[0], c_xx, c_length, &c_basisveccopy[0], c_length, c_Nm, c_index, &c_pairing_new[0], &c_onsite_new[0],c_half)
#	cdef int i
#	cdef int j
#	for i in range(c_xx):
#		vectorfree_matrix_productinitial(&c_length, &c_val[0],&c_indx[0],&c_pntrb[0], &c_pntre[0],&c_basisvec[c_length*i],&c_basisveccopy[c_length*i])
#		c_onsite_new[c_Nm*i]=1
#		c_onsite_new[c_Nm*i+1]=c_basisveccopy[c_length*i+c_process*c_xx+i+c_half]
#		c_pairing_new[c_Nm*i]=0
#		c_pairing_new[c_Nm*i+1]=c_basisveccopy[c_length*i+c_process*c_xx+i]
#		#timeinitial=time.time()
#		for j in range (1,c_Nmhalf):
#			vectorfree_matrix_product(&c_length, &c_val[0],&c_indx[0],&c_pntrb[0], &c_pntre[0],&c_basisveccopy[c_length*i],&c_basisvec[c_length*i])
#			vectorfree_matrix_product(&c_length, &c_val[0],&c_indx[0],&c_pntrb[0], &c_pntre[0],&c_basisvec[c_length*i],&c_basisveccopy[c_length*i])
#			#timefinal=time.time()
#			c_onsite_new[c_Nm*i+2*j]=c_basisvec[c_length*i+c_process*c_xx+i+c_half]
#			c_onsite_new[c_Nm*i+2*j+1]=c_basisveccopy[c_length*i+c_process*c_xx+i+c_half]
#			c_pairing_new[c_Nm*i+2*j]=c_basisvec[c_length*i+c_process*c_xx+i]
#			c_pairing_new[c_Nm*i+2*j+1]=c_basisveccopy[c_length*i+c_process*c_xx+i]
#	#time2=time.time()
#	#print(time2-time1)
#	#timefinal=time.time()
#	#print(timefinal-timeinitial)			
	return



