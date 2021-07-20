#include "sparse_d_mm_chebyshev_new.h"
#include "stdio.h"


void sparse_d_mm_chebyshev(double *AVal, MKL_INT *AColInd, MKL_INT *ARowPtrB, MKL_INT *ARowPtrE, double* x, MKL_INT columns, MKL_INT ldx, double* y, MKL_INT ldy, MKL_INT niter, MKL_INT index, double* momentspairing, double* momentsonsite,MKL_INT sitenumber){
	// csrA:(ldy)x(ldx), x:=(ldx)x(columns), y:=(ldy)x(columns)
	// alpha*csrA*x+beta*y
	// ldx equivalent to martrixsites(2*number of sites)
	// ldx should be equal to ldy in our case
	// columns equivalent to nvectors
	double* tmp;
	struct matrix_descr descrA;
	sparse_matrix_t csrA; 
	sparse_status_t status;
	//start chebyshev	
	double alpha=1.0;
	double beta=0.0;
//	mkl_set_num_threads(4);
	descrA.type=SPARSE_MATRIX_TYPE_GENERAL;
	status=mkl_sparse_d_create_csr(&csrA,SPARSE_INDEX_BASE_ZERO,ldy,ldx,ARowPtrB,ARowPtrE,AColInd,AVal);
	//mkl_sparse_optimize (csrA);
	status=mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE,alpha,csrA,descrA,SPARSE_LAYOUT_COLUMN_MAJOR,x,columns,ldx,beta,y,ldy);
	//for (long long j=0;j<columns*ldx;j++){
	//	printf("%f ",y[j]);
	//}
	//printf("\n");
	for(int j=0; j<columns; j++){
		momentsonsite[j*niter]=1;
		//momentsonsite[j*niter+1]=y[index+j+j*ldx+sitenumber];
		momentsonsite[j*niter+1]=y[index+j+j*ldx+sitenumber];
		momentspairing[j*niter]=0;
		momentspairing[j*niter+1]=y[index+j+j*ldx];
	}
	tmp=x;
	x=y;
	y=tmp;
	// hier beginnt normale Chebyshev-Entwicklung
	alpha=2.0;
	beta=-1.0;

	status=mkl_sparse_d_create_csr(&csrA,SPARSE_INDEX_BASE_ZERO,ldy,ldx,ARowPtrB,ARowPtrE,AColInd,AVal);
	mkl_sparse_optimize (csrA);
	for(int i=2; i<niter; i++){
		status=mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE,alpha,csrA,descrA,SPARSE_LAYOUT_COLUMN_MAJOR,x,columns,ldx,beta,y,ldy);
		for(int j=0; j<columns; j++){
			momentsonsite[i+j*niter]=y[index+j+j*ldx+sitenumber];
			momentspairing[i+j*niter]=y[index+j+j*ldx];
		}
		tmp=x;
		x=y;
		y=tmp;
	}
}
