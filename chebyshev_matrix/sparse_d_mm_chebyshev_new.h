#include "mkl_spblas.h"
#include "mkl.h"

// -lsparse_d_mm_chebyshev_new

#ifndef SPBLAS_H
#define SPBLAS_H


void sparse_d_mm_chebyshev(double *AVal, MKL_INT *AColInd, MKL_INT *ARowPtrB, MKL_INT *ARowPtrE, double* x, MKL_INT columns, MKL_INT ldx, double* y, MKL_INT ldy, MKL_INT niter, MKL_INT index, double* momentspairing, double* momentsonsite,MKL_INT sitenumber);
#endif
