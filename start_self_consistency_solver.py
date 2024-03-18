#!/usr/bin/env python2
import sys
sys.path.append('./Modules')
from self_consistency_MatrixFree_bdg import *
import cy_chebyshev_cython_double

if __name__=="__main__":
        initial_time=time.time()

        dtype=np.float64

        #n is the filling factor
        n=np.float64(0.3)

        #W is the disorder strength
        Warr=[100, 250, 400]

        #U is the interaction strength
        Uarr=[220]

        #T is the temperature
        Tarr={}
        Tarr[0.5]={}
        Tarr[1.0]={}
        Tarr[2.5]={}
        Tarr[4.0]={}    
        Tarr[7.0]={}    
#       Tarr[10.0]={}   

        Tarr[0.5][2.2]=np.array([0.0, 0.03, 0.06])
        Tarr[0.5][1.8]=np.array([0.0, 0.012, 0.024])
        Tarr[0.5][1.5]=np.array([0.0, 0.004, 0.008])

        Tarr[1.0][2.2]=np.array([0.0, 0.04, 0.08])
        Tarr[1.0][1.8]=np.array([0.0, 0.014, 0.028])
        Tarr[1.0][1.5]=np.array([0.0, 0.005, 0.01])
        
        Tarr[2.5][2.2]=np.array([0.0, 0.06, 0.12])
        Tarr[2.5][1.8]=np.array([0.0, 0.025, 0.05])
        Tarr[2.5][1.5]=np.array([0.0, 0.01, 0.02])
        
        Tarr[4.0][2.2]=np.array([0.0, 0.08, 0.16])
        Tarr[4.0][1.8]=np.array([0.0, 0.03, 0.06])
        Tarr[4.0][1.5]=np.array([0.0, 0.015, 0.03])

        Tarr[7.0][2.2]=np.array([0.0, 0.1, 0.2])
        Tarr[7.0][1.8]=np.array([0.0, 0.035, 0.07])
        Tarr[7.0][1.5]=np.array([0.0, 0.02, 0.04])
        
#       Tarr[10.0][2.2]=np.array([0.0, 0.01, 0.02, 0.03, 0.05, 0.065, 0.12, 0.15])
#       Tarr[10.0][1.8]=np.array([0.0, 0.04, 0.08, 0.012, 0.018, 0.024, 0.048, 0.062])
#       Tarr[10.0][1.5]=np.array([0.0, 0.01, 0.02])

        #Nm is the number of Chebyshev moments
        Nmarr={}
        
        for U in Uarr:
                U=dtype(round(0.01*U,2))
                Nmarr[U]={}
                if U!=1.5:
                        for W in Warr:
                                W=dtype(round(0.01*W,2))
                                Nmarr[U][W]={}
                                for iT in range(1):
                                        Nmarr[U][W][round(Tarr[W][U][iT],3)]=[1024, 2048, 8192]
                                for iT in range(1,Tarr[W][U].size):
                                        Nmarr[U][W][round(Tarr[W][U][iT],3)]=[2048, 4096, 8192]
                else:
                        for W in Warr:
                                W=dtype(round(0.01*W,2))
                                Nmarr[U][W]={}
                                for iT in range(1):
                                        Nmarr[U][W][round(Tarr[W][U][iT],3)]=[2048, 4096]
                                for iT in range(1,Tarr[W][U].size):
                                        Nmarr[U][W][round(Tarr[W][U][iT],3)]=[4096, 8192, 16384]

        tolerance=0.001

        argarr=sys.argv

        if len(argarr)<=3:
            print("Not enough arguments!")
            exit(1)

        directory=argarr[1]
        fnumber=int(argarr[2])
        L=int(argarr[3])

        comm=MPI.COMM_WORLD
        mpi_rank=comm.Get_rank()

        niter_max=10001
        niter=10000
        njobs=1

        seed_RndVec=0
        nconfigs=1

        diag=False

        for iconfig in range(nconfigs):
                if len(argarr)==5:
                        name='bdg_'+str(mpi_rank+int(argarr[4])+iconfig*njobs*fnumber)+'.hdf5'
                        if name not in listdir(directory):
                                with h5py.File(directory+'/{0}'.format(name),'w-') as f:
                                    pass
                        if int(mpi_rank)==0:
                                print(name)
                else:
                        name='bdg_'+str(mpi_rank+fnumber+iconfig*500)+'.hdf5'
                        while True:
                                count=0
                                try:
                                        with h5py.File(directory+'/{0}'.format(name),'w-') as f:
                                                pass
                                        break
                                except IOError:
                                        count=count+1
                                        name='bdg_'+str(mpi_rank+fnumber+count+iconfig)+'.hdf5'
                for W in Warr: 
                        W=dtype(round(0.01*W,2))
                        for U in Uarr:
                                U=dtype(round(0.01*U,2))
                                for T in Tarr[W][U]:
                                        T=dtype(round(T, 6))
                                        for Nm in Nmarr[U][W][T]:
#                                               print('T', T, 'W', W, 'U', U, 'Nm', Nm)
                                                if Nm:
                                                    diag=False
                                                else:
                                                    diag=True
                                                delta_i, n_i, mu, iterstart, self_consistency, cycle_time, seed, deleteNm = get_start(directory+'/'+name, n, T, W, U, Nm, L, delta_start=4.3*np.exp(-5./U), deleteNm=False)
                                                # set seed
                                                cryptogen = SystemRandom()                                      
                                                if seed is None:
                                                        seed=cryptogen.getrandbits(32)
                                                self_consistency=np.array(self_consistency)
                                                self_consistency_tmp=np.zeros((niter_max,3))
                                                self_consistency_tmp[:self_consistency.shape[0],:]=self_consistency[:,:]
                                                self_consistency=self_consistency_tmp
                                                if all(abs(self_con)<=tolerance for self_con in self_consistency[iterstart,:]):
                                                        continue
                                                endflag=self_consistency_loop(L=L, n=n, T=T, Nm=Nm, W=W, U=U, niter=niter, path=directory+'/'+name, tolerance=tolerance, dtype=dtype, seed=seed, initial_time=initial_time, delta_i=delta_i, n_i=n_i, mu=mu, iterstart=iterstart, self_consistency=self_consistency, cycle_time=cycle_time, diag=diag)
                                                if endflag==1:
                                                        break
