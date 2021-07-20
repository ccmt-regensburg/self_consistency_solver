#include "chebyshev_cython_double.h"
#include<stdio.h>
#include<malloc.h>
#include<math.h>

#define N 96 
#define NrndVec 28

void mychebyshev(double* restrict onsite, double* restrict pairing, double* restrict randvec, int randvec_nonzero, int Nmoments, double hopping, double* restrict onsite_new, double* restrict pairing_new)
{
	int i,m,n;
	double* restrict tmpvec1=_mm_malloc(2*N*N*NrndVec*sizeof(double), 64);
	double* restrict tmpvec2=_mm_malloc(2*N*N*NrndVec*sizeof(double), 64);	
	double* restrict tmpvec3=_mm_malloc(2*N*N*NrndVec*sizeof(double), 64);

	__assume_aligned(onsite, 64);
	__assume_aligned(pairing, 64);
	__assume_aligned(onsite_new, 64);
	__assume_aligned(pairing_new, 64);
	#pragma omp parallel for private(i,n) num_threads(NrndVec)
	for(n=0; n<NrndVec; n++){
		#pragma simd
		for(i=N+1; i<N*N-N-1; i++){
			tmpvec1[i+2*n*N*N]=onsite[i]*randvec[i+2*n*N*N]+pairing[i]*randvec[i+2*n*N*N+N*N]-hopping*randvec[i-1+2*n*N*N]-hopping*randvec[i+1+2*n*N*N]-hopping*randvec[i-N+2*n*N*N]-hopping*randvec[i+N+2*n*N*N];
			tmpvec1[i+2*n*N*N+N*N]=pairing[i]*randvec[i+2*n*N*N]-onsite[i]*randvec[i+2*n*N*N+N*N]+hopping*randvec[i-1+2*n*N*N+N*N]+hopping*randvec[i+1+2*n*N*N+N*N]+hopping*randvec[i-N+2*n*N*N+N*N]+hopping*randvec[i+N+2*n*N*N+N*N];
		}
		tmpvec1[2*n*N*N]=onsite[0]*randvec[2*n*N*N]+pairing[0]*randvec[2*n*N*N+N*N]-hopping*randvec[(N-1)+2*n*N*N]-hopping*randvec[1+2*n*N*N]-hopping*randvec[N*N-N+2*n*N*N]-hopping*randvec[N+2*n*N*N];
		tmpvec1[2*n*N*N+N*N]=pairing[0]*randvec[2*n*N*N]-onsite[0]*randvec[2*n*N*N+N*N]+hopping*randvec[N-1+2*n*N*N+N*N]+hopping*randvec[1+2*n*N*N+N*N]+hopping*randvec[N*N-N+2*n*N*N+N*N]+hopping*randvec[N+2*n*N*N+N*N];	

		tmpvec1[(N-1)+2*n*N*N]=onsite[(N-1)]*randvec[(N-1)+2*n*N*N]+pairing[(N-1)]*randvec[(N-1)+2*n*N*N+N*N]-hopping*randvec[(N-1)-1+2*n*N*N]-hopping*randvec[2*n*N*N]-hopping*randvec[(N*N-1)+2*n*N*N]-hopping*randvec[(N-1)+N+2*n*N*N];
		tmpvec1[(N-1)+2*n*N*N+N*N]=pairing[(N-1)]*randvec[(N-1)+2*n*N*N]-onsite[(N-1)]*randvec[(N-1)+2*n*N*N+N*N]+hopping*randvec[(N-1)-1+2*n*N*N+N*N]+hopping*randvec[2*n*N*N+N*N]+hopping*randvec[(N*N-1)+2*n*N*N+N*N]+hopping*randvec[(N-1)+N+2*n*N*N+N*N];

		tmpvec1[(N*N-N)+2*n*N*N]=onsite[(N*N-N)]*randvec[(N*N-N)+2*n*N*N]+pairing[(N*N-N)]*randvec[(N*N-N)+2*n*N*N+N*N]-hopping*randvec[(N*N-1)+2*n*N*N]-hopping*randvec[(N*N-N)+1+2*n*N*N]-hopping*randvec[(N*N-N)-N+2*n*N*N]-hopping*randvec[2*n*N*N];
		tmpvec1[(N*N-N)+2*n*N*N+N*N]=pairing[(N*N-N)]*randvec[(N*N-N)+2*n*N*N]-onsite[(N*N-N)]*randvec[(N*N-N)+2*n*N*N+N*N]+hopping*randvec[(N*N-1)+2*n*N*N+N*N]+hopping*randvec[(N*N-N)+1+2*n*N*N+N*N]+hopping*randvec[(N*N-N)-N+2*n*N*N+N*N]+hopping*randvec[2*n*N*N+N*N];

		tmpvec1[(N*N-1)+2*n*N*N]=onsite[(N*N-1)]*randvec[(N*N-1)+2*n*N*N]+pairing[(N*N-1)]*randvec[(N*N-1)+2*n*N*N+N*N]-hopping*randvec[(N*N-1)-1+2*n*N*N]-hopping*randvec[(N*N-N)+2*n*N*N]-hopping*randvec[(N-1)+2*n*N*N]-hopping*randvec[(N*N-1)-N+2*n*N*N];
		tmpvec1[(N*N-1)+2*n*N*N+N*N]=pairing[(N*N-1)]*randvec[(N*N-1)+2*n*N*N]-onsite[(N*N-1)]*randvec[(N*N-1)+2*n*N*N+N*N]+hopping*randvec[(N*N-1)-1+2*n*N*N+N*N]+hopping*randvec[(N*N-N)+2*n*N*N+N*N]+hopping*randvec[(N-1)+2*n*N*N+N*N]+hopping*randvec[(N*N-1)-N+2*n*N*N+N*N];
		#pragma simd
		for(i=1; i<N-1; i++){
			tmpvec1[i+2*n*N*N]=onsite[i]*randvec[i+2*n*N*N]+pairing[i]*randvec[i+2*n*N*N+N*N]-hopping*randvec[i-1+2*n*N*N]-hopping*randvec[i+1+2*n*N*N]-hopping*randvec[i+N*N-N+2*n*N*N]-hopping*randvec[i+N+2*n*N*N];
			tmpvec1[i+2*n*N*N+N*N]=pairing[i]*randvec[i+2*n*N*N]-onsite[i]*randvec[i+2*n*N*N+N*N]+hopping*randvec[i-1+2*n*N*N+N*N]+hopping*randvec[i+1+2*n*N*N+N*N]+hopping*randvec[i+N*N-N+2*n*N*N+N*N]+hopping*randvec[i+N+2*n*N*N+N*N];	

			tmpvec1[i*N+2*n*N*N]=onsite[i*N]*randvec[i*N+2*n*N*N]+pairing[i*N]*randvec[i*N+2*n*N*N+N*N]-hopping*randvec[i*N+(N-1)+2*n*N*N]-hopping*randvec[i*N+1+2*n*N*N]-hopping*randvec[i*N-N+2*n*N*N]-hopping*randvec[i*N+N+2*n*N*N];
			tmpvec1[i*N+2*n*N*N+N*N]=pairing[i*N]*randvec[i*N+2*n*N*N]-onsite[i*N]*randvec[i*N+2*n*N*N+N*N]+hopping*randvec[i*N+(N-1)+2*n*N*N+N*N]+hopping*randvec[i*N+1+2*n*N*N+N*N]+hopping*randvec[i*N-N+2*n*N*N+N*N]+hopping*randvec[i*N+N+2*n*N*N+N*N];

			tmpvec1[i*N+(N-1)+2*n*N*N]=onsite[i*N+(N-1)]*randvec[i*N+(N-1)+2*n*N*N]+pairing[i*N+(N-1)]*randvec[i*N+(N-1)+2*n*N*N+N*N]-hopping*randvec[i*N+(N-1)-1+2*n*N*N]-hopping*randvec[i*N+2*n*N*N]-hopping*randvec[i*N+(N-1)-N+2*n*N*N]-hopping*randvec[i*N+(N-1)+N+2*n*N*N];
			tmpvec1[i*N+(N-1)+2*n*N*N+N*N]=pairing[i*N+(N-1)]*randvec[i*N+(N-1)+2*n*N*N]-onsite[i*N+(N-1)]*randvec[i*N+(N-1)+2*n*N*N+N*N]+hopping*randvec[i*N+(N-1)-1+2*n*N*N+N*N]+hopping*randvec[i*N+2*n*N*N+N*N]+hopping*randvec[i*N+(N-1)-N+2*n*N*N+N*N]+hopping*randvec[i*N+(N-1)+N+2*n*N*N+N*N];

			tmpvec1[(N*N-N)+i+2*n*N*N]=onsite[(N*N-N)+i]*randvec[(N*N-N)+i+2*n*N*N]+pairing[(N*N-N)+i]*randvec[(N*N-N)+i+2*n*N*N+N*N]-hopping*randvec[(N*N-N)+i-1+2*n*N*N]-hopping*randvec[(N*N-N)+i+1+2*n*N*N]-hopping*randvec[(N*N-N)+i-N+2*n*N*N]-hopping*randvec[i+2*n*N*N];
			tmpvec1[(N*N-N)+i+2*n*N*N+N*N]=pairing[(N*N-N)+i]*randvec[(N*N-N)+i+2*n*N*N]-onsite[(N*N-N)+i]*randvec[(N*N-N)+i+2*n*N*N+N*N]+hopping*randvec[(N*N-N)+i-1+2*n*N*N+N*N]+hopping*randvec[(N*N-N)+i+1+2*n*N*N+N*N]+hopping*randvec[(N*N-N)+i-N+2*n*N*N+N*N]+hopping*randvec[i+2*n*N*N+N*N];
		}
	}
	for(n=0; n<NrndVec; n++){
		onsite_new[n*Nmoments+randvec_nonzero*Nmoments]=randvec[randvec_nonzero+n+2*n*N*N];
		pairing_new[n*Nmoments+randvec_nonzero*Nmoments]=randvec[randvec_nonzero+n+N*N+2*n*N*N];
 	}
	for(n=0; n<NrndVec; n++){
		onsite_new[1+n*Nmoments+randvec_nonzero*Nmoments]=tmpvec1[randvec_nonzero+n+2*n*N*N];
		pairing_new[1+n*Nmoments+randvec_nonzero*Nmoments]=tmpvec1[randvec_nonzero+n+N*N+2*n*N*N];
 	}
	#pragma omp parallel for private(i,n) num_threads(NrndVec)
	for(n=0; n<NrndVec; n++){
		#pragma simd
		for(i=N+1; i<N*N-N-1; i++){
			tmpvec2[i+2*n*N*N]=((onsite[i]*tmpvec1[i+2*n*N*N]+pairing[i]*tmpvec1[i+2*n*N*N+N*N]-hopping*tmpvec1[i-1+2*n*N*N]-hopping*tmpvec1[i+1+2*n*N*N]-hopping*tmpvec1[i-N+2*n*N*N]-hopping*tmpvec1[i+N+2*n*N*N])*2.-randvec[i+2*n*N*N]);
			tmpvec2[i+2*n*N*N+N*N]=((pairing[i]*tmpvec1[i+2*n*N*N]-onsite[i]*tmpvec1[i+2*n*N*N+N*N]+hopping*tmpvec1[i-1+2*n*N*N+N*N]+hopping*tmpvec1[i+1+2*n*N*N+N*N]+hopping*tmpvec1[i-N+2*n*N*N+N*N]+hopping*tmpvec1[i+N+2*n*N*N+N*N])*2.-randvec[i+2*n*N*N+N*N]);
		}
		tmpvec2[2*n*N*N]=((onsite[0]*tmpvec1[2*n*N*N]+pairing[0]*tmpvec1[2*n*N*N+N*N]-hopping*tmpvec1[(N-1)+2*n*N*N]-hopping*tmpvec1[1+2*n*N*N]-hopping*tmpvec1[N*N-N+2*n*N*N]-hopping*tmpvec1[N+2*n*N*N])*2.-randvec[2*n*N*N]);
		tmpvec2[2*n*N*N+N*N]=((pairing[0]*tmpvec1[2*n*N*N]-onsite[0]*tmpvec1[2*n*N*N+N*N]+hopping*tmpvec1[N-1+2*n*N*N+N*N]+hopping*tmpvec1[1+2*n*N*N+N*N]+hopping*tmpvec1[N*N-N+2*n*N*N+N*N]+hopping*tmpvec1[N+2*n*N*N+N*N])*2.-randvec[2*n*N*N+N*N]);	

		tmpvec2[(N-1)+2*n*N*N]=((onsite[(N-1)]*tmpvec1[(N-1)+2*n*N*N]+pairing[(N-1)]*tmpvec1[(N-1)+2*n*N*N+N*N]-hopping*tmpvec1[(N-1)-1+2*n*N*N]-hopping*tmpvec1[2*n*N*N]-hopping*tmpvec1[(N*N-1)+2*n*N*N]-hopping*tmpvec1[(N-1)+N+2*n*N*N])*2.-randvec[(N-1)+2*n*N*N]);
		tmpvec2[(N-1)+2*n*N*N+N*N]=((pairing[(N-1)]*tmpvec1[(N-1)+2*n*N*N]-onsite[(N-1)]*tmpvec1[(N-1)+2*n*N*N+N*N]+hopping*tmpvec1[(N-1)-1+2*n*N*N+N*N]+hopping*tmpvec1[2*n*N*N+N*N]+hopping*tmpvec1[(N*N-1)+2*n*N*N+N*N]+hopping*tmpvec1[(N-1)+N+2*n*N*N+N*N])*2.-randvec[(N-1)+2*n*N*N+N*N]);

		tmpvec2[(N*N-N)+2*n*N*N]=((onsite[(N*N-N)]*tmpvec1[(N*N-N)+2*n*N*N]+pairing[(N*N-N)]*tmpvec1[(N*N-N)+2*n*N*N+N*N]-hopping*tmpvec1[(N*N-1)+2*n*N*N]-hopping*tmpvec1[(N*N-N)+1+2*n*N*N]-hopping*tmpvec1[(N*N-N)-N+2*n*N*N]-hopping*tmpvec1[2*n*N*N])*2.-randvec[(N*N-N)+2*n*N*N]);
		tmpvec2[(N*N-N)+2*n*N*N+N*N]=((pairing[(N*N-N)]*tmpvec1[(N*N-N)+2*n*N*N]-onsite[(N*N-N)]*tmpvec1[(N*N-N)+2*n*N*N+N*N]+hopping*tmpvec1[(N*N-1)+2*n*N*N+N*N]+hopping*tmpvec1[(N*N-N)+1+2*n*N*N+N*N]+hopping*tmpvec1[(N*N-N)-N+2*n*N*N+N*N]+hopping*tmpvec1[2*n*N*N+N*N])*2.-randvec[(N*N-N)+2*n*N*N+N*N]);

		tmpvec2[(N*N-1)+2*n*N*N]=((onsite[(N*N-1)]*tmpvec1[(N*N-1)+2*n*N*N]+pairing[(N*N-1)]*tmpvec1[(N*N-1)+2*n*N*N+N*N]-hopping*tmpvec1[(N*N-1)-1+2*n*N*N]-hopping*tmpvec1[(N*N-N)+2*n*N*N]-hopping*tmpvec1[(N-1)+2*n*N*N]-hopping*tmpvec1[(N*N-1)-N+2*n*N*N])*2.-randvec[(N*N-1)+2*n*N*N]);
		tmpvec2[(N*N-1)+2*n*N*N+N*N]=((pairing[(N*N-1)]*tmpvec1[(N*N-1)+2*n*N*N]-onsite[(N*N-1)]*tmpvec1[(N*N-1)+2*n*N*N+N*N]+hopping*tmpvec1[(N*N-1)-1+2*n*N*N+N*N]+hopping*tmpvec1[(N*N-N)+2*n*N*N+N*N]+hopping*tmpvec1[(N-1)+2*n*N*N+N*N]+hopping*tmpvec1[(N*N-1)-N+2*n*N*N+N*N])*2.-randvec[(N*N-1)+2*n*N*N+N*N]);

		#pragma simd
		for(i=1; i<N-1; i++){
			tmpvec2[i+2*n*N*N]=((onsite[i]*tmpvec1[i+2*n*N*N]+pairing[i]*tmpvec1[i+2*n*N*N+N*N]-hopping*tmpvec1[i-1+2*n*N*N]-hopping*tmpvec1[i+1+2*n*N*N]-hopping*tmpvec1[i+N*N-N+2*n*N*N]-hopping*tmpvec1[i+N+2*n*N*N])*2.-randvec[i+2*n*N*N]);
			tmpvec2[i+2*n*N*N+N*N]=((pairing[i]*tmpvec1[i+2*n*N*N]-onsite[i]*tmpvec1[i+2*n*N*N+N*N]+hopping*tmpvec1[i-1+2*n*N*N+N*N]+hopping*tmpvec1[i+1+2*n*N*N+N*N]+hopping*tmpvec1[i+N*N-N+2*n*N*N+N*N]+hopping*tmpvec1[i+N+2*n*N*N+N*N])*2.-randvec[i+2*n*N*N+N*N]);

			tmpvec2[i*N+2*n*N*N]=((onsite[i*N]*tmpvec1[i*N+2*n*N*N]+pairing[i*N]*tmpvec1[i*N+2*n*N*N+N*N]-hopping*tmpvec1[i*N+(N-1)+2*n*N*N]-hopping*tmpvec1[i*N+1+2*n*N*N]-hopping*tmpvec1[i*N-N+2*n*N*N]-hopping*tmpvec1[i*N+N+2*n*N*N])*2.-randvec[i*N+2*n*N*N]);
			tmpvec2[i*N+2*n*N*N+N*N]=((pairing[i*N]*tmpvec1[i*N+2*n*N*N]-onsite[i*N]*tmpvec1[i*N+2*n*N*N+N*N]+hopping*tmpvec1[i*N+(N-1)+2*n*N*N+N*N]+hopping*tmpvec1[i*N+1+2*n*N*N+N*N]+hopping*tmpvec1[i*N-N+2*n*N*N+N*N]+hopping*tmpvec1[i*N+N+2*n*N*N+N*N])*2.-randvec[i*N+2*n*N*N+N*N]);

			tmpvec2[i*N+(N-1)+2*n*N*N]=((onsite[i*N+(N-1)]*tmpvec1[i*N+(N-1)+2*n*N*N]+pairing[i*N+(N-1)]*tmpvec1[i*N+(N-1)+2*n*N*N+N*N]	-hopping*tmpvec1[i*N+(N-1)-1+2*n*N*N]-hopping*tmpvec1[i*N+2*n*N*N]-hopping*tmpvec1[i*N+(N-1)-N+2*n*N*N]-hopping*tmpvec1[i*N+(N-1)+N+2*n*N*N])*2.-randvec[i*N+(N-1)+2*n*N*N]);
			tmpvec2[i*N+(N-1)+2*n*N*N+N*N]=((pairing[i*N+(N-1)]*tmpvec1[i*N+(N-1)+2*n*N*N]-onsite[i*N+(N-1)]*tmpvec1[i*N+(N-1)+2*n*N*N+N*N]+hopping*tmpvec1[i*N+(N-1)-1+2*n*N*N+N*N]+hopping*tmpvec1[i*N+2*n*N*N+N*N]+hopping*tmpvec1[i*N+(N-1)-N+2*n*N*N+N*N]+hopping*tmpvec1[i*N+(N-1)+N+2*n*N*N+N*N])*2.-randvec[i*N+(N-1)+2*n*N*N+N*N]);

			tmpvec2[(N*N-N)+i+2*n*N*N]=((onsite[(N*N-N)+i]*tmpvec1[(N*N-N)+i+2*n*N*N]+pairing[(N*N-N)+i]*tmpvec1[(N*N-N)+i+2*n*N*N+N*N]-hopping*tmpvec1[(N*N-N)+i-1+2*n*N*N]-hopping*tmpvec1[(N*N-N)+i+1+2*n*N*N]-hopping*tmpvec1[(N*N-N)+i-N+2*n*N*N]-hopping*tmpvec1[i+2*n*N*N])*2.-randvec[(N*N-N)+i+2*n*N*N]);
			tmpvec2[(N*N-N)+i+2*n*N*N+N*N]=((pairing[(N*N-N)+i]*tmpvec1[(N*N-N)+i+2*n*N*N]-onsite[(N*N-N)+i]*tmpvec1[(N*N-N)+i+2*n*N*N+N*N]+hopping*tmpvec1[(N*N-N)+i-1+2*n*N*N+N*N]+hopping*tmpvec1[(N*N-N)+i+1+2*n*N*N+N*N]+hopping*tmpvec1[(N*N-N)+i-N+2*n*N*N+N*N]+hopping*tmpvec1[i+2*n*N*N+N*N])*2.-randvec[(N*N-N)+i+2*n*N*N+N*N]);
		}
	}			
	for(n=0; n<NrndVec; n++){
		onsite_new[2+n*Nmoments+randvec_nonzero*Nmoments]=tmpvec2[randvec_nonzero+n+2*n*N*N];
		pairing_new[2+n*Nmoments+randvec_nonzero*Nmoments]=tmpvec2[randvec_nonzero+n+N*N+2*n*N*N];
 	}
	for(m=3; m<Nmoments; m++){	
		if(m%3==0){
			#pragma omp parallel for private(i,n) num_threads(NrndVec)
			for(n=0; n<NrndVec; n++){
				#pragma simd
				for(i=N+1; i<N*N-N-1; i++){
					tmpvec3[i+2*n*N*N]=((onsite[i]*tmpvec2[i+2*n*N*N]+pairing[i]*tmpvec2[i+2*n*N*N+N*N]-hopping*tmpvec2[i-1+2*n*N*N]-hopping*tmpvec2[i+1+2*n*N*N]-hopping*tmpvec2[i-N+2*n*N*N]-hopping*tmpvec2[i+N+2*n*N*N])*2.-tmpvec1[i+2*n*N*N]);
					tmpvec3[i+2*n*N*N+N*N]=((pairing[i]*tmpvec2[i+2*n*N*N]-onsite[i]*tmpvec2[i+2*n*N*N+N*N]+hopping*tmpvec2[i-1+2*n*N*N+N*N]+hopping*tmpvec2[i+1+2*n*N*N+N*N]+hopping*tmpvec2[i-N+2*n*N*N+N*N]+hopping*tmpvec2[i+N+2*n*N*N+N*N])*2.-tmpvec1[i+2*n*N*N+N*N]);
				}
				tmpvec3[2*n*N*N]=((onsite[0]*tmpvec2[2*n*N*N]+pairing[0]*tmpvec2[2*n*N*N+N*N]-hopping*tmpvec2[(N-1)+2*n*N*N]-hopping*tmpvec2[1+2*n*N*N]-hopping*tmpvec2[N*N-N+2*n*N*N]-hopping*tmpvec2[N+2*n*N*N])*2.-tmpvec1[2*n*N*N]);
				tmpvec3[2*n*N*N+N*N]=((pairing[0]*tmpvec2[2*n*N*N]-onsite[0]*tmpvec2[2*n*N*N+N*N]+hopping*tmpvec2[N-1+2*n*N*N+N*N]+hopping*tmpvec2[1+2*n*N*N+N*N]+hopping*tmpvec2[N*N-N+2*n*N*N+N*N]+hopping*tmpvec2[N+2*n*N*N+N*N])*2.-tmpvec1[2*n*N*N+N*N]);	

				tmpvec3[(N-1)+2*n*N*N]=((onsite[(N-1)]*tmpvec2[(N-1)+2*n*N*N]+pairing[(N-1)]*tmpvec2[(N-1)+2*n*N*N+N*N]-hopping*tmpvec2[(N-1)-1+2*n*N*N]-hopping*tmpvec2[2*n*N*N]-hopping*tmpvec2[(N*N-1)+2*n*N*N]-hopping*tmpvec2[(N-1)+N+2*n*N*N])*2.-tmpvec1[(N-1)+2*n*N*N]);
				tmpvec3[(N-1)+2*n*N*N+N*N]=((pairing[(N-1)]*tmpvec2[(N-1)+2*n*N*N]-onsite[(N-1)]*tmpvec2[(N-1)+2*n*N*N+N*N]+hopping*tmpvec2[(N-1)-1+2*n*N*N+N*N]+hopping*tmpvec2[2*n*N*N+N*N]+hopping*tmpvec2[(N*N-1)+2*n*N*N+N*N]+hopping*tmpvec2[(N-1)+N+2*n*N*N+N*N])*2.-tmpvec1[(N-1)+2*n*N*N+N*N]);

				tmpvec3[(N*N-N)+2*n*N*N]=((onsite[(N*N-N)]*tmpvec2[(N*N-N)+2*n*N*N]+pairing[(N*N-N)]*tmpvec2[(N*N-N)+2*n*N*N+N*N]-hopping*tmpvec2[(N*N-1)+2*n*N*N]-hopping*tmpvec2[(N*N-N)+1+2*n*N*N]-hopping*tmpvec2[(N*N-N)-N+2*n*N*N]-hopping*tmpvec2[2*n*N*N])*2.-tmpvec1[(N*N-N)+2*n*N*N]);
				tmpvec3[(N*N-N)+2*n*N*N+N*N]=((pairing[(N*N-N)]*tmpvec2[(N*N-N)+2*n*N*N]-onsite[(N*N-N)]*tmpvec2[(N*N-N)+2*n*N*N+N*N]+hopping*tmpvec2[(N*N-1)+2*n*N*N+N*N]+hopping*tmpvec2[(N*N-N)+1+2*n*N*N+N*N]+hopping*tmpvec2[(N*N-N)-N+2*n*N*N+N*N]+hopping*tmpvec2[2*n*N*N+N*N])*2.-tmpvec1[(N*N-N)+2*n*N*N+N*N]);

				tmpvec3[(N*N-1)+2*n*N*N]=((onsite[(N*N-1)]*tmpvec2[(N*N-1)+2*n*N*N]+pairing[(N*N-1)]*tmpvec2[(N*N-1)+2*n*N*N+N*N]-hopping*tmpvec2[(N*N-1)-1+2*n*N*N]-hopping*tmpvec2[(N*N-N)+2*n*N*N]-hopping*tmpvec2[(N-1)+2*n*N*N]-hopping*tmpvec2[(N*N-1)-N+2*n*N*N])*2.-tmpvec1[(N*N-1)+2*n*N*N]);
				tmpvec3[(N*N-1)+2*n*N*N+N*N]=((pairing[(N*N-1)]*tmpvec2[(N*N-1)+2*n*N*N]-onsite[(N*N-1)]*tmpvec2[(N*N-1)+2*n*N*N+N*N]+hopping*tmpvec2[(N*N-1)-1+2*n*N*N+N*N]+hopping*tmpvec2[(N*N-N)+2*n*N*N+N*N]+hopping*tmpvec2[(N-1)+2*n*N*N+N*N]+hopping*tmpvec2[(N*N-1)-N+2*n*N*N+N*N])*2.-tmpvec1[(N*N-1)+2*n*N*N+N*N]);

				#pragma simd
				for(i=1; i<N-1; i++){
					tmpvec3[i+2*n*N*N]=((onsite[i]*tmpvec2[i+2*n*N*N]+pairing[i]*tmpvec2[i+2*n*N*N+N*N]-hopping*tmpvec2[i-1+2*n*N*N]-hopping*tmpvec2[i+1+2*n*N*N]-hopping*tmpvec2[i+N*N-N+2*n*N*N]-hopping*tmpvec2[i+N+2*n*N*N])*2.-tmpvec1[i+2*n*N*N]);
					tmpvec3[i+2*n*N*N+N*N]=((pairing[i]*tmpvec2[i+2*n*N*N]-onsite[i]*tmpvec2[i+2*n*N*N+N*N]+hopping*tmpvec2[i-1+2*n*N*N+N*N]+hopping*tmpvec2[i+1+2*n*N*N+N*N]+hopping*tmpvec2[i+N*N-N+2*n*N*N+N*N]+hopping*tmpvec2[i+N+2*n*N*N+N*N])*2.-tmpvec1[i+2*n*N*N+N*N]);

					tmpvec3[i*N+2*n*N*N]=((onsite[i*N]*tmpvec2[i*N+2*n*N*N]+pairing[i*N]*tmpvec2[i*N+2*n*N*N+N*N]-hopping*tmpvec2[i*N+(N-1)+2*n*N*N]-hopping*tmpvec2[i*N+1+2*n*N*N]-hopping*tmpvec2[i*N-N+2*n*N*N]-hopping*tmpvec2[i*N+N+2*n*N*N])*2.-tmpvec1[i*N+2*n*N*N]);
					tmpvec3[i*N+2*n*N*N+N*N]=((pairing[i*N]*tmpvec2[i*N+2*n*N*N]-onsite[i*N]*tmpvec2[i*N+2*n*N*N+N*N]+hopping*tmpvec2[i*N+(N-1)+2*n*N*N+N*N]+hopping*tmpvec2[i*N+1+2*n*N*N+N*N]+hopping*tmpvec2[i*N-N+2*n*N*N+N*N]+hopping*tmpvec2[i*N+N+2*n*N*N+N*N])*2.-tmpvec1[i*N+2*n*N*N+N*N]);

					tmpvec3[i*N+(N-1)+2*n*N*N]=((onsite[i*N+(N-1)]*tmpvec2[i*N+(N-1)+2*n*N*N]+pairing[i*N+(N-1)]*tmpvec2[i*N+(N-1)+2*n*N*N+N*N]	-hopping*tmpvec2[i*N+(N-1)-1+2*n*N*N]-hopping*tmpvec2[i*N+2*n*N*N]-hopping*tmpvec2[i*N+(N-1)-N+2*n*N*N]-hopping*tmpvec2[i*N+(N-1)+N+2*n*N*N])*2.-tmpvec1[i*N+(N-1)+2*n*N*N]);
					tmpvec3[i*N+(N-1)+2*n*N*N+N*N]=((pairing[i*N+(N-1)]*tmpvec2[i*N+(N-1)+2*n*N*N]-onsite[i*N+(N-1)]*tmpvec2[i*N+(N-1)+2*n*N*N+N*N]+hopping*tmpvec2[i*N+(N-1)-1+2*n*N*N+N*N]+hopping*tmpvec2[i*N+2*n*N*N+N*N]+hopping*tmpvec2[i*N+(N-1)-N+2*n*N*N+N*N]+hopping*tmpvec2[i*N+(N-1)+N+2*n*N*N+N*N])*2.-tmpvec1[i*N+(N-1)+2*n*N*N+N*N]);

					tmpvec3[(N*N-N)+i+2*n*N*N]=((onsite[(N*N-N)+i]*tmpvec2[(N*N-N)+i+2*n*N*N]+pairing[(N*N-N)+i]*tmpvec2[(N*N-N)+i+2*n*N*N+N*N]-hopping*tmpvec2[(N*N-N)+i-1+2*n*N*N]-hopping*tmpvec2[(N*N-N)+i+1+2*n*N*N]-hopping*tmpvec2[(N*N-N)+i-N+2*n*N*N]-hopping*tmpvec2[i+2*n*N*N])*2.-tmpvec1[(N*N-N)+i+2*n*N*N]);
					tmpvec3[(N*N-N)+i+2*n*N*N+N*N]=((pairing[(N*N-N)+i]*tmpvec2[(N*N-N)+i+2*n*N*N]-onsite[(N*N-N)+i]*tmpvec2[(N*N-N)+i+2*n*N*N+N*N]+hopping*tmpvec2[(N*N-N)+i-1+2*n*N*N+N*N]+hopping*tmpvec2[(N*N-N)+i+1+2*n*N*N+N*N]+hopping*tmpvec2[(N*N-N)+i-N+2*n*N*N+N*N]+hopping*tmpvec2[i+2*n*N*N+N*N])*2.-tmpvec1[(N*N-N)+i+2*n*N*N+N*N]);
				}
			}			
			for(n=0; n<NrndVec; n++){
				onsite_new[m+n*Nmoments+randvec_nonzero*Nmoments]=tmpvec3[randvec_nonzero+n+2*n*N*N];
				pairing_new[m+n*Nmoments+randvec_nonzero*Nmoments]=tmpvec3[randvec_nonzero+n+N*N+2*n*N*N];
 			}
		}
		else{
			if(m%3==1){				
				#pragma omp parallel for private(i,n) num_threads(NrndVec)
				for(n=0; n<NrndVec; n++){
					#pragma simd
					for(i=N+1; i<N*N-N-1; i++){
						tmpvec1[i+2*n*N*N]=((onsite[i]*tmpvec3[i+2*n*N*N]+pairing[i]*tmpvec3[i+2*n*N*N+N*N]-hopping*tmpvec3[i-1+2*n*N*N]-hopping*tmpvec3[i+1+2*n*N*N]-hopping*tmpvec3[i-N+2*n*N*N]-hopping*tmpvec3[i+N+2*n*N*N])*2.-tmpvec2[i+2*n*N*N]);
						tmpvec1[i+2*n*N*N+N*N]=((pairing[i]*tmpvec3[i+2*n*N*N]-onsite[i]*tmpvec3[i+2*n*N*N+N*N]+hopping*tmpvec3[i-1+2*n*N*N+N*N]+hopping*tmpvec3[i+1+2*n*N*N+N*N]+hopping*tmpvec3[i-N+2*n*N*N+N*N]+hopping*tmpvec3[i+N+2*n*N*N+N*N])*2.-tmpvec2[i+2*n*N*N+N*N]);
					}
					tmpvec1[2*n*N*N]=((onsite[0]*tmpvec3[2*n*N*N]+pairing[0]*tmpvec3[2*n*N*N+N*N]-hopping*tmpvec3[(N-1)+2*n*N*N]-hopping*tmpvec3[1+2*n*N*N]-hopping*tmpvec3[N*N-N+2*n*N*N]-hopping*tmpvec3[N+2*n*N*N])*2.-tmpvec2[2*n*N*N]);
					tmpvec1[2*n*N*N+N*N]=((pairing[0]*tmpvec3[2*n*N*N]-onsite[0]*tmpvec3[2*n*N*N+N*N]+hopping*tmpvec3[N-1+2*n*N*N+N*N]+hopping*tmpvec3[1+2*n*N*N+N*N]+hopping*tmpvec3[N*N-N+2*n*N*N+N*N]+hopping*tmpvec3[N+2*n*N*N+N*N])*2.-tmpvec2[2*n*N*N+N*N]);	

					tmpvec1[(N-1)+2*n*N*N]=((onsite[(N-1)]*tmpvec3[(N-1)+2*n*N*N]+pairing[(N-1)]*tmpvec3[(N-1)+2*n*N*N+N*N]-hopping*tmpvec3[(N-1)-1+2*n*N*N]-hopping*tmpvec3[2*n*N*N]-hopping*tmpvec3[(N*N-1)+2*n*N*N]-hopping*tmpvec3[(N-1)+N+2*n*N*N])*2.-tmpvec2[(N-1)+2*n*N*N]);
					tmpvec1[(N-1)+2*n*N*N+N*N]=((pairing[(N-1)]*tmpvec3[(N-1)+2*n*N*N]-onsite[(N-1)]*tmpvec3[(N-1)+2*n*N*N+N*N]+hopping*tmpvec3[(N-1)-1+2*n*N*N+N*N]+hopping*tmpvec3[2*n*N*N+N*N]+hopping*tmpvec3[(N*N-1)+2*n*N*N+N*N]+hopping*tmpvec3[(N-1)+N+2*n*N*N+N*N])*2.-tmpvec2[(N-1)+2*n*N*N+N*N]);

					tmpvec1[(N*N-N)+2*n*N*N]=((onsite[(N*N-N)]*tmpvec3[(N*N-N)+2*n*N*N]+pairing[(N*N-N)]*tmpvec3[(N*N-N)+2*n*N*N+N*N]-hopping*tmpvec3[(N*N-1)+2*n*N*N]-hopping*tmpvec3[(N*N-N)+1+2*n*N*N]-hopping*tmpvec3[(N*N-N)-N+2*n*N*N]-hopping*tmpvec3[2*n*N*N])*2.-tmpvec2[(N*N-N)+2*n*N*N]);
					tmpvec1[(N*N-N)+2*n*N*N+N*N]=((pairing[(N*N-N)]*tmpvec3[(N*N-N)+2*n*N*N]-onsite[(N*N-N)]*tmpvec3[(N*N-N)+2*n*N*N+N*N]+hopping*tmpvec3[(N*N-1)+2*n*N*N+N*N]+hopping*tmpvec3[(N*N-N)+1+2*n*N*N+N*N]+hopping*tmpvec3[(N*N-N)-N+2*n*N*N+N*N]+hopping*tmpvec3[2*n*N*N+N*N])*2.-tmpvec2[(N*N-N)+2*n*N*N+N*N]);

					tmpvec1[(N*N-1)+2*n*N*N]=((onsite[(N*N-1)]*tmpvec3[(N*N-1)+2*n*N*N]+pairing[(N*N-1)]*tmpvec3[(N*N-1)+2*n*N*N+N*N]-hopping*tmpvec3[(N*N-1)-1+2*n*N*N]-hopping*tmpvec3[(N*N-N)+2*n*N*N]-hopping*tmpvec3[(N-1)+2*n*N*N]-hopping*tmpvec3[(N*N-1)-N+2*n*N*N])*2.-tmpvec2[(N*N-1)+2*n*N*N]);
					tmpvec1[(N*N-1)+2*n*N*N+N*N]=((pairing[(N*N-1)]*tmpvec3[(N*N-1)+2*n*N*N]-onsite[(N*N-1)]*tmpvec3[(N*N-1)+2*n*N*N+N*N]+hopping*tmpvec3[(N*N-1)-1+2*n*N*N+N*N]+hopping*tmpvec3[(N*N-N)+2*n*N*N+N*N]+hopping*tmpvec3[(N-1)+2*n*N*N+N*N]+hopping*tmpvec3[(N*N-1)-N+2*n*N*N+N*N])*2.-tmpvec2[(N*N-1)+2*n*N*N+N*N]);

					#pragma simd
					for(i=1; i<N-1; i++){
						tmpvec1[i+2*n*N*N]=((onsite[i]*tmpvec3[i+2*n*N*N]+pairing[i]*tmpvec3[i+2*n*N*N+N*N]-hopping*tmpvec3[i-1+2*n*N*N]-hopping*tmpvec3[i+1+2*n*N*N]-hopping*tmpvec3[i+N*N-N+2*n*N*N]-hopping*tmpvec3[i+N+2*n*N*N])*2.-tmpvec2[i+2*n*N*N]);
						tmpvec1[i+2*n*N*N+N*N]=((pairing[i]*tmpvec3[i+2*n*N*N]-onsite[i]*tmpvec3[i+2*n*N*N+N*N]+hopping*tmpvec3[i-1+2*n*N*N+N*N]+hopping*tmpvec3[i+1+2*n*N*N+N*N]+hopping*tmpvec3[i+N*N-N+2*n*N*N+N*N]+hopping*tmpvec3[i+N+2*n*N*N+N*N])*2.-tmpvec2[i+2*n*N*N+N*N]);

						tmpvec1[i*N+2*n*N*N]=((onsite[i*N]*tmpvec3[i*N+2*n*N*N]+pairing[i*N]*tmpvec3[i*N+2*n*N*N+N*N]-hopping*tmpvec3[i*N+(N-1)+2*n*N*N]-hopping*tmpvec3[i*N+1+2*n*N*N]-hopping*tmpvec3[i*N-N+2*n*N*N]-hopping*tmpvec3[i*N+N+2*n*N*N])*2.-tmpvec2[i*N+2*n*N*N]);
						tmpvec1[i*N+2*n*N*N+N*N]=((pairing[i*N]*tmpvec3[i*N+2*n*N*N]-onsite[i*N]*tmpvec3[i*N+2*n*N*N+N*N]	+hopping*tmpvec3[i*N+(N-1)+2*n*N*N+N*N]+hopping*tmpvec3[i*N+1+2*n*N*N+N*N]+hopping*tmpvec3[i*N-N+2*n*N*N+N*N]+hopping*tmpvec3[i*N+N+2*n*N*N+N*N])*2.-tmpvec2[i*N+2*n*N*N+N*N]);

						tmpvec1[i*N+(N-1)+2*n*N*N]=((onsite[i*N+(N-1)]*tmpvec3[i*N+(N-1)+2*n*N*N]+pairing[i*N+(N-1)]*tmpvec3[i*N+(N-1)+2*n*N*N+N*N]-hopping*tmpvec3[i*N+(N-1)-1+2*n*N*N]-hopping*tmpvec3[i*N+2*n*N*N]-hopping*tmpvec3[i*N+(N-1)-N+2*n*N*N]-hopping*tmpvec3[i*N+(N-1)+N+2*n*N*N])*2.-tmpvec2[i*N+(N-1)+2*n*N*N]);
						tmpvec1[i*N+(N-1)+2*n*N*N+N*N]=((pairing[i*N+(N-1)]*tmpvec3[i*N+(N-1)+2*n*N*N]-onsite[i*N+(N-1)]*tmpvec3[i*N+(N-1)+2*n*N*N+N*N]+hopping*tmpvec3[i*N+(N-1)-1+2*n*N*N+N*N]+hopping*tmpvec3[i*N+2*n*N*N+N*N]+hopping*tmpvec3[i*N+(N-1)-N+2*n*N*N+N*N]+hopping*tmpvec3[i*N+(N-1)+N+2*n*N*N+N*N])*2.-tmpvec2[i*N+(N-1)+2*n*N*N+N*N]);

						tmpvec1[(N*N-N)+i+2*n*N*N]=((onsite[(N*N-N)+i]*tmpvec3[(N*N-N)+i+2*n*N*N]+pairing[(N*N-N)+i]*tmpvec3[(N*N-N)+i+2*n*N*N+N*N]-hopping*tmpvec3[(N*N-N)+i-1+2*n*N*N]-hopping*tmpvec3[(N*N-N)+i+1+2*n*N*N]-hopping*tmpvec3[(N*N-N)+i-N+2*n*N*N]-hopping*tmpvec3[i+2*n*N*N])*2.-tmpvec2[(N*N-N)+i+2*n*N*N]);
						tmpvec1[(N*N-N)+i+2*n*N*N+N*N]=((pairing[(N*N-N)+i]*tmpvec3[(N*N-N)+i+2*n*N*N]-onsite[(N*N-N)+i]*tmpvec3[(N*N-N)+i+2*n*N*N+N*N]+hopping*tmpvec3[(N*N-N)+i-1+2*n*N*N+N*N]+hopping*tmpvec3[(N*N-N)+i+1+2*n*N*N+N*N]+hopping*tmpvec3[(N*N-N)+i-N+2*n*N*N+N*N]+hopping*tmpvec3[i+2*n*N*N+N*N])*2.-tmpvec2[(N*N-N)+i+2*n*N*N+N*N]);
					}
				}
				for(n=0; n<NrndVec; n++){
					onsite_new[m+n*Nmoments+randvec_nonzero*Nmoments]=tmpvec1[randvec_nonzero+n+2*n*N*N];
					pairing_new[m+n*Nmoments+randvec_nonzero*Nmoments]=tmpvec1[randvec_nonzero+n+N*N+2*n*N*N];
 				}
			}
			else{
				#pragma omp parallel for private(i,n) num_threads(NrndVec)
				for(n=0; n<NrndVec; n++){
					#pragma simd
					for(i=N+1; i<N*N-N-1; i++){
						tmpvec2[i+2*n*N*N]=((onsite[i]*tmpvec1[i+2*n*N*N]+pairing[i]*tmpvec1[i+2*n*N*N+N*N]-hopping*tmpvec1[i-1+2*n*N*N]-hopping*tmpvec1[i+1+2*n*N*N]-hopping*tmpvec1[i-N+2*n*N*N]-hopping*tmpvec1[i+N+2*n*N*N])*2.-tmpvec3[i+2*n*N*N]);
						tmpvec2[i+2*n*N*N+N*N]=((pairing[i]*tmpvec1[i+2*n*N*N]-onsite[i]*tmpvec1[i+2*n*N*N+N*N]+hopping*tmpvec1[i-1+2*n*N*N+N*N]+hopping*tmpvec1[i+1+2*n*N*N+N*N]+hopping*tmpvec1[i-N+2*n*N*N+N*N]+hopping*tmpvec1[i+N+2*n*N*N+N*N])*2.-tmpvec3[i+2*n*N*N+N*N]);
					}
					tmpvec2[2*n*N*N]=((onsite[0]*tmpvec1[2*n*N*N]+pairing[0]*tmpvec1[2*n*N*N+N*N]-hopping*tmpvec1[(N-1)+2*n*N*N]-hopping*tmpvec1[1+2*n*N*N]-hopping*tmpvec1[N*N-N+2*n*N*N]-hopping*tmpvec1[N+2*n*N*N])*2.-tmpvec3[2*n*N*N]);
					tmpvec2[2*n*N*N+N*N]=((pairing[0]*tmpvec1[2*n*N*N]-onsite[0]*tmpvec1[2*n*N*N+N*N]+hopping*tmpvec1[N-1+2*n*N*N+N*N]+hopping*tmpvec1[1+2*n*N*N+N*N]+hopping*tmpvec1[N*N-N+2*n*N*N+N*N]+hopping*tmpvec1[N+2*n*N*N+N*N])*2.-tmpvec3[2*n*N*N+N*N]);	

					tmpvec2[(N-1)+2*n*N*N]=((onsite[(N-1)]*tmpvec1[(N-1)+2*n*N*N]+pairing[(N-1)]*tmpvec1[(N-1)+2*n*N*N+N*N]-hopping*tmpvec1[(N-1)-1+2*n*N*N]-hopping*tmpvec1[2*n*N*N]-hopping*tmpvec1[(N*N-1)+2*n*N*N]-hopping*tmpvec1[(N-1)+N+2*n*N*N])*2.-tmpvec3[(N-1)+2*n*N*N]);
					tmpvec2[(N-1)+2*n*N*N+N*N]=((pairing[(N-1)]*tmpvec1[(N-1)+2*n*N*N]-onsite[(N-1)]*tmpvec1[(N-1)+2*n*N*N+N*N]+hopping*tmpvec1[(N-1)-1+2*n*N*N+N*N]+hopping*tmpvec1[2*n*N*N+N*N]+hopping*tmpvec1[(N*N-1)+2*n*N*N+N*N]+hopping*tmpvec1[(N-1)+N+2*n*N*N+N*N])*2.-tmpvec3[(N-1)+2*n*N*N+N*N]);

					tmpvec2[(N*N-N)+2*n*N*N]=((onsite[(N*N-N)]*tmpvec1[(N*N-N)+2*n*N*N]+pairing[(N*N-N)]*tmpvec1[(N*N-N)+2*n*N*N+N*N]-hopping*tmpvec1[(N*N-1)+2*n*N*N]-hopping*tmpvec1[(N*N-N)+1+2*n*N*N]-hopping*tmpvec1[(N*N-N)-N+2*n*N*N]-hopping*tmpvec1[2*n*N*N])*2.-tmpvec3[(N*N-N)+2*n*N*N]);
					tmpvec2[(N*N-N)+2*n*N*N+N*N]=((pairing[(N*N-N)]*tmpvec1[(N*N-N)+2*n*N*N]-onsite[(N*N-N)]*tmpvec1[(N*N-N)+2*n*N*N+N*N]+hopping*tmpvec1[(N*N-1)+2*n*N*N+N*N]+hopping*tmpvec1[(N*N-N)+1+2*n*N*N+N*N]+hopping*tmpvec1[(N*N-N)-N+2*n*N*N+N*N]+hopping*tmpvec1[2*n*N*N+N*N])*2.-tmpvec3[(N*N-N)+2*n*N*N+N*N]);

					tmpvec2[(N*N-1)+2*n*N*N]=((onsite[(N*N-1)]*tmpvec1[(N*N-1)+2*n*N*N]+pairing[(N*N-1)]*tmpvec1[(N*N-1)+2*n*N*N+N*N]-hopping*tmpvec1[(N*N-1)-1+2*n*N*N]-hopping*tmpvec1[(N*N-N)+2*n*N*N]-hopping*tmpvec1[(N-1)+2*n*N*N]-hopping*tmpvec1[(N*N-1)-N+2*n*N*N])*2.-tmpvec3[(N*N-1)+2*n*N*N]);
					tmpvec2[(N*N-1)+2*n*N*N+N*N]=((pairing[(N*N-1)]*tmpvec1[(N*N-1)+2*n*N*N]-onsite[(N*N-1)]*tmpvec1[(N*N-1)+2*n*N*N+N*N]+hopping*tmpvec1[(N*N-1)-1+2*n*N*N+N*N]+hopping*tmpvec1[(N*N-N)+2*n*N*N+N*N]+hopping*tmpvec1[(N-1)+2*n*N*N+N*N]+hopping*tmpvec1[(N*N-1)-N+2*n*N*N+N*N])*2.-tmpvec3[(N*N-1)+2*n*N*N+N*N]);

					#pragma simd
					for(i=1; i<N-1; i++){
						tmpvec2[i+2*n*N*N]=((onsite[i]*tmpvec1[i+2*n*N*N]+pairing[i]*tmpvec1[i+2*n*N*N+N*N]-hopping*tmpvec1[i-1+2*n*N*N]-hopping*tmpvec1[i+1+2*n*N*N]-hopping*tmpvec1[i+N*N-N+2*n*N*N]-hopping*tmpvec1[i+N+2*n*N*N])*2.-tmpvec3[i+2*n*N*N]);
						tmpvec2[i+2*n*N*N+N*N]=((pairing[i]*tmpvec1[i+2*n*N*N]-onsite[i]*tmpvec1[i+2*n*N*N+N*N]+hopping*tmpvec1[i-1+2*n*N*N+N*N]+hopping*tmpvec1[i+1+2*n*N*N+N*N]+hopping*tmpvec1[i+N*N-N+2*n*N*N+N*N]+hopping*tmpvec1[i+N+2*n*N*N+N*N])*2.-tmpvec3[i+2*n*N*N+N*N]);

						tmpvec2[i*N+2*n*N*N]=((onsite[i*N]*tmpvec1[i*N+2*n*N*N]+pairing[i*N]*tmpvec1[i*N+2*n*N*N+N*N]-hopping*tmpvec1[i*N+(N-1)+2*n*N*N]-hopping*tmpvec1[i*N+1+2*n*N*N]-hopping*tmpvec1[i*N-N+2*n*N*N]-hopping*tmpvec1[i*N+N+2*n*N*N])*2.-tmpvec3[i*N+2*n*N*N]);
						tmpvec2[i*N+2*n*N*N+N*N]=((pairing[i*N]*tmpvec1[i*N+2*n*N*N]-onsite[i*N]*tmpvec1[i*N+2*n*N*N+N*N]+hopping*tmpvec1[i*N+(N-1)+2*n*N*N+N*N]+hopping*tmpvec1[i*N+1+2*n*N*N+N*N]+hopping*tmpvec1[i*N-N+2*n*N*N+N*N]+hopping*tmpvec1[i*N+N+2*n*N*N+N*N])*2.-tmpvec3[i*N+2*n*N*N+N*N]);

						tmpvec2[i*N+(N-1)+2*n*N*N]=((onsite[i*N+(N-1)]*tmpvec1[i*N+(N-1)+2*n*N*N]+pairing[i*N+(N-1)]*tmpvec1[i*N+(N-1)+2*n*N*N+N*N]-hopping*tmpvec1[i*N+(N-1)-1+2*n*N*N]-hopping*tmpvec1[i*N+2*n*N*N]-hopping*tmpvec1[i*N+(N-1)-N+2*n*N*N]-hopping*tmpvec1[i*N+(N-1)+N+2*n*N*N])*2.-tmpvec3[i*N+(N-1)+2*n*N*N]);
						tmpvec2[i*N+(N-1)+2*n*N*N+N*N]=((pairing[i*N+(N-1)]*tmpvec1[i*N+(N-1)+2*n*N*N]-onsite[i*N+(N-1)]*tmpvec1[i*N+(N-1)+2*n*N*N+N*N]+hopping*tmpvec1[i*N+(N-1)-1+2*n*N*N+N*N]+hopping*tmpvec1[i*N+2*n*N*N+N*N]+hopping*tmpvec1[i*N+(N-1)-N+2*n*N*N+N*N]+hopping*tmpvec1[i*N+(N-1)+N+2*n*N*N+N*N])*2.-tmpvec3[i*N+(N-1)+2*n*N*N+N*N]);

						tmpvec2[(N*N-N)+i+2*n*N*N]=((onsite[(N*N-N)+i]*tmpvec1[(N*N-N)+i+2*n*N*N]+pairing[(N*N-N)+i]*tmpvec1[(N*N-N)+i+2*n*N*N+N*N]-hopping*tmpvec1[(N*N-N)+i-1+2*n*N*N]-hopping*tmpvec1[(N*N-N)+i+1+2*n*N*N]-hopping*tmpvec1[(N*N-N)+i-N+2*n*N*N]-hopping*tmpvec1[i+2*n*N*N])*2.-tmpvec3[(N*N-N)+i+2*n*N*N]);
						tmpvec2[(N*N-N)+i+2*n*N*N+N*N]=((pairing[(N*N-N)+i]*tmpvec1[(N*N-N)+i+2*n*N*N]-onsite[(N*N-N)+i]*tmpvec1[(N*N-N)+i+2*n*N*N+N*N]+hopping*tmpvec1[(N*N-N)+i-1+2*n*N*N+N*N]+hopping*tmpvec1[(N*N-N)+i+1+2*n*N*N+N*N]+hopping*tmpvec1[(N*N-N)+i-N+2*n*N*N+N*N]+hopping*tmpvec1[i+2*n*N*N+N*N])*2.-tmpvec3[(N*N-N)+i+2*n*N*N+N*N]);
					}
				}
				for(n=0; n<NrndVec; n++){
					onsite_new[m+n*Nmoments+randvec_nonzero*Nmoments]=tmpvec2[randvec_nonzero+n+2*n*N*N];
					pairing_new[m+n*Nmoments+randvec_nonzero*Nmoments]=tmpvec2[randvec_nonzero+n+N*N+2*n*N*N];
 				}
			}
		}
	}
	_mm_free(tmpvec1);
	_mm_free(tmpvec2);
	_mm_free(tmpvec3);
}
