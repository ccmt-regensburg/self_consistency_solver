#!/usr/bin/env python2

from __future__ import division
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg.eigen.arpack as arp
import scipy.fftpack as fftpack
import time
from random import SystemRandom
from os import listdir
import sys
import h5py
from mpi4py import MPI
import resource
import cy_chebyshev_cython_double

#number of threads used in Chebyshev expansion
chebynRnd=20
#wall time
timelimit=48

def fermi(beta, E):
        """This function returns the value of the fermi function with fermi energy 0 at energy E.
        Arguments:
                beta: inverse thermal energy
                E: energy value
        Return salues:
                fermi function value
        """
        return 1./(1.+np.exp(E*beta))


def initialize_file(path, nRndVec, n, T, W, U, Nm, seed, delta_start, n_start, iterstart, dtype):
        """This function initializes the format and data of a HDF5 file.
        Arguments:
                path: path to file
                nlat: number of lattice sites
                n: particle density
                T: temperature
                W: disorder strength
                U: interaction strength
                Nm: number of Chebyshev moments
                tol: targetted self-consistency tolerance
                seed: disorder configuration seed
                niter: maximum number of iterations
                delta_i: local pairing amplitude
                n_i: local occupation number
                cycle_time: expended time of all iterations
                dtype: data type
        Return values:
                cycle_time: already expended time in previous iterations
                
        """

        roundto=6
        
        with h5py.File(path, 'r+') as f:
                nRVstr='nRV_{}'.format(nRndVec)
                nstr='n_'+str(round(n,roundto))
                Tstr='T_'+str(round(T,2*roundto))
                Wstr='W_'+str(round(W,roundto))
                Ustr='U_'+str(round(U,roundto))
                Nmstr='Nm_'+str(Nm)
                if 'seed' not in f.attrs:       
                        f.attrs['seed']=seed
                if 'maxrss' not in f.attrs:
                        f.attrs['maxrss']=-1.
                if nRVstr not in f:
                        f.create_group(nRVstr)
                if nstr not in f[nRVstr]:
                        f[nRVstr].create_group(nstr)
                if Tstr not in f[nRVstr][nstr]:
                        f[nRVstr][nstr].create_group(Tstr)
                if Wstr not in f[nRVstr][nstr][Tstr]:    
                        f[nRVstr][nstr][Tstr].create_group(Wstr)
                if Ustr not in f[nRVstr][nstr][Tstr][Wstr]:
                        f[nRVstr][nstr][Tstr][Wstr].create_group(Ustr)
                if Nmstr not in f[nRVstr][nstr][Tstr][Wstr][Ustr]:
                        f[nRVstr][nstr][Tstr][Wstr][Ustr].create_group(Nmstr)
                if iterstart==0:
                        if 'delta_start' not in f[nRVstr][nstr][Tstr][Wstr][Ustr][Nmstr]:
                                f[nRVstr][nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr].create_dataset('delta_start', delta_start.shape, dtype, data=delta_start)
                        if 'n_start' not in f[nRVstr][nstr][Tstr][Wstr][Ustr][Nmstr]:
                                f[nRVstr][nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr].create_dataset('n_start', n_start.shape, dtype, data=n_start)      

def write_data(nRndVec,path, nlat, n, T, W, U, Nm, tol, niter, delta_i, n_i, mu, iterations, cycle_time, self_consistency, maxrss, dos, pairingdos, emax, e, dtype=np.float64):
        """This function writes the data computed in a self-consistency iterations to a HDF5 file.
        Arguments:
                path: path to file
                nlat: number of lattice sites
                n: particle density
                T: temperature
                W: disorder strength
                U: interaction strength
                Nm: number of Chebyshev moments
                tol: targetted self-consistency tolerance
                seed: disorder configuration seed
                niter: maximum number of iterations
                delta_i: local pairing amplitude
                n_i: local occupation number
                mu: chemical potential
                iterations: number of executed iterations in self-consistency cycle
                cycle_time: expended time of all iterations
                self_consistency: tolerance up to which self-consistency has been achieved
                maxrss: maximum resident set size (maximum RAM usage)
                dtype: data type
        """
        roundto=6
        with h5py.File(path, 'r+') as f:
                nRVstr='nRV_{}'.format(nRndVec)
                nstr='n_'+str(round(n,roundto))
                Tstr='T_'+str(round(T,2*roundto))
                Wstr='W_'+str(round(W,roundto))
                Ustr='U_'+str(round(U,roundto))
                Nmstr='Nm_'+str(Nm)
                tolstr='tol_'+str(round(tol,roundto))
                if dtype(f.attrs['maxrss'])<maxrss:
                        f.attrs['maxrss']=maxrss
                if tolstr not in f[nRVstr+'/'+nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr]:
                        f[nRVstr+'/'+nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr].create_group(tolstr) 
                if 'n_i' not in f[nRVstr+'/'+nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr+'/'+tolstr]:
                        f[nRVstr+'/'+nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr+'/'+tolstr].create_dataset('n_i', n_i.shape, dtype, data=n_i)
                else:
                        f[nRVstr+'/'+nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr+'/'+tolstr]['n_i'][:]=n_i[:]
                if 'delta_i' not in f[nRVstr+'/'+nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr+'/'+tolstr]:
                        f[nRVstr+'/'+nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr+'/'+tolstr].create_dataset('delta_i', delta_i.shape, dtype, data=delta_i)
                else:
                        f[nRVstr+'/'+nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr+'/'+tolstr]['delta_i'][:]=delta_i[:]
                if 'dos' not in f[nRVstr+'/'+nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr+'/'+tolstr]:
                        f[nRVstr+'/'+nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr+'/'+tolstr].create_dataset('dos', dos.shape, dtype, data=dos)
                else:
                        f[nRVstr+'/'+nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr+'/'+tolstr]['dos'][:]=dos[:]
                if 'pairingdos' not in f[nRVstr+'/'+nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr+'/'+tolstr]:
                        f[nRVstr+'/'+nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr+'/'+tolstr].create_dataset('pairingdos', pairingdos.shape, dtype, data=pairingdos)
                else:
                        f[nRVstr+'/'+nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr+'/'+tolstr]['pairingdos'][:]=pairingdos[:]
                if e is not False:
                        if 'e' not in f[nRVstr+'/'+nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr+'/'+tolstr]:
                                f[nRVstr+'/'+nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr+'/'+tolstr].create_dataset('e', e.shape, dtype, data=e)
                        else:
                                f[nRVstr+'/'+nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr+'/'+tolstr]['e'][:]=e[:] 
                f[nRVstr+'/'+nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr+'/'+tolstr].attrs['delta']=dtype(np.sum(delta_i)/nlat)
                f[nRVstr+'/'+nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr+'/'+tolstr].attrs['iterations']=np.int(iterations)
                f[nRVstr+'/'+nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr+'/'+tolstr].attrs['mu']=dtype(mu)
                f[nRVstr+'/'+nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr+'/'+tolstr].attrs['emax']=dtype(emax)
                f[nRVstr+'/'+nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr].attrs['time']=cycle_time
                if 'self_consistency' not in f[nRVstr+'/'+nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr]:
                        f[nRVstr+'/'+nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr].create_dataset('self_consistency', data=self_consistency)
                else:
                        if f[nRVstr+'/'+nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr]['self_consistency'].shape!=self_consistency.shape:
                                if (f[nRVstr+'/'+nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr]['self_consistency'].shape[0]<self_consistency.shape[0] 
                                    or f[nRVstr+'/'+nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr]['self_consistency'].shape[1]!=self_consistency.shape[1]):
                                        del f[nRVstr+'/'+nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr]['self_consistency']
                                        f[nRVstr+'/'+nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr].create_dataset('self_consistency', data=self_consistency)
                        else:
                                f[nRVstr+'/'+nstr+'/'+Tstr+'/'+Wstr+'/'+Ustr+'/'+Nmstr]['self_consistency'][:,:]=self_consistency[:,:]
        return

def get_start(path, n, T, W, U, Nm, L, delta_start=0.12, mustart=-0.85, dtype=np.float64, tol=0.01, mode=['', 'Nm', 'T', 'U', 'W'], deleteNm=False):
        """This function fetches the data necessary to start the self-consistency cycle from a HDF5 file.
        Arguments:
                path: path to file
                n: particle density 
                T: temperature
                W: disorder strength
                U: interaction strength
                Nm: number of moments used for Chebyshev expansion
                L: linear size of the lattice
                ndensity: electron density
        Keyword Arguments:
                delta_start: value of the homogeneous starting guess for the pairing amplitude if no appropriate data could be found
                mustart: starting chemical potential
                dtype: data type of the returned data
                tol: tolerance up to which self-consistency is assumed
                mode: order of preffered starting configuration (see function get_data)
        Return Values:
                delta_i: local pairing amplitude
                n_i: local occupation number
                iterations: number of iterations that have already been performed (only relevant for mode '')
                self_con: tolerance up to which self-consistency has been achieved (only relevant for mode '')
                seed: seed of the disorder configuration corresponding to the file 
        """
        def get_data(f, mode=''):
                """This function returns the data corresponding to the given parameter configuration that is necessary to restart the self-consistency cycle from a HDF5 file.
                Arguments:
                        n: particle density
                        T: temperature
                        W: disorder strength
                        U: interaction strength
                        Nm: number of moments in Chebyshev expansion
                        f: file handle
                Keyword Arguments:
                        mode: determines how n, T, W, U and Nm are interpreted to get the configuration. 'n', 'T', 'W' and 'U' modes can be combined.
                                '': Check for specific configuration W, U, T
                                'n', 'T', 'W', 'U': Check for closest configuration with smaller n, W, U or T
                Return Values:
                        delta_i: local pairing amplitude
                        n_i: local occupation number
                        iterations: number of iterations already performed (only applicable to standard mode '')
                        self_con: Tolerance up to which self-consistency has been reached. None is returned if the configuration does not exist. Inf signifies an undetermined tolerance.       
                """

                def closest_key(key, dtype, f, noself=True, smallereq=False):
                        """This function returns the closest key with a smaller value in a HDF5 file.
                        Arguments:
                                key: key of which the closest smaller-valued key is searched for (key format: 'Name_Value')
                                dtype: data type of the value of the key
                                f: file handle of the HDF5 file
                        Return Values:
                        closest_key: closest smaller key
                        """
                        closest_key=''
                        keyname, keyval = key.split('_')
                        keyval=np.abs(dtype(keyval))
                        keylist=f.keys()
                        if noself==1:
                                keylist=np.array([dtype(keyel.split('_')[1]) for keyel in keylist if keyname==keyel.split('_')[0] and not np.allclose(dtype(keyel.split('_')[1]), keyval)])
                        else:   
                                keylist=np.array([dtype(keyel.split('_')[1]) for keyel in keylist if keyname==keyel.split('_')[0]])
                        if smallereq==1:
                                keylist=keylist[np.where((keylist-keyval)<=0)]
                        if keylist.size:
                                closest_key=keyname+'_'+str(keylist[np.argmin(np.abs(keylist-keyval))])
                        else:
                                closest_key=key
                        return closest_key 
                def checkkey(key, pathprefix, pathsuffix, dtype, f, tolerance=0.1):
                        """This function checks if using the smaller-valued key starting configuration is feasible. It is called, when "c" is included in the mode.
                        Arguments:
                                key: key of which the closest smaller-valued key is searched for (key format: 'Name_Value')
                                pathprefix: prefix of the path where to compare key
                                pathsuffix: suffix of the path where to compare key
                                dtype: data type of the value of the key
                                f: file handle of the HDF5 file
                        Keyword Arguments:
                                tolerance: tolerance of the smallest 
                        Return Values:
                                True/False                              
                        """
                        check=False
                        if pathprefix not in f:
                                return check
                        closekey=closest_key(key, dtype, f[pathprefix], noself=True)
                        delta_i_key=None
                        delta_i_closekey=None
                        if key in f[pathprefix]:
                                if 'self_consistency' in f[pathprefix+key+pathsuffix].attrs:
                                        self_con=f[pathprefix+key+pathsuffix].attrs['self_consistency']
                                        if isinstance(self_con, float):
                                                tolstr='tol_'+str(round(self_con, roundto))
                                        else:
                                                if isinstance(self_con[-1], float):
                                                        tolstr='tol_'+str(round(self_con[-1], roundto))
                                                else:
                                                        tolstr='tol_'+str(round(self_con[-1][0], roundto))
                                        tolstr=closest_key(tolstr, np.float64, f[pathprefix+key+pathsuffix], noself=False, smallereq=True)
                                        if tolstr not in f[pathprefix+key+pathsuffix]:
                                                tolstr=closest_key(tolstr, np.float64, f[pathprefix+key+pathsuffix])
                                        if tolstr in f[pathprefix+key+pathsuffix]:
                                                if 'delta_i' in f[pathprefix+key+pathsuffix+'/'+tolstr]:
                                                        delta_i_key=f[pathprefix+key+pathsuffix+'/'+tolstr]['delta_i'][:]
                                if 'self_consistency' in f[pathprefix+key+pathsuffix]:
                                        self_con=f[pathprefix+key+pathsuffix]['self_consistency'][:,:]
                                        tolstr='tol_'+str(round((self_con[np.nonzero(self_con[:,0]),0])[0,-1], roundto))

                                        tolstr=closest_key(tolstr, np.float64, f[pathprefix+key+pathsuffix], noself=False, smallereq=True)
                                        if tolstr not in f[pathprefix+key+pathsuffix]:
                                                tolstr=closest_key(tolstr, np.float64, f[pathprefix+key+pathsuffix])
                                        if tolstr in f[pathprefix+key+pathsuffix]:
                                                if 'delta_i' in f[pathprefix+key+pathsuffix+'/'+tolstr]:
                                                        delta_i_key=f[pathprefix+key+pathsuffix+'/'+tolstr]['delta_i'][:]
                                if 'self_consistency' in f[pathprefix+closekey+pathsuffix].attrs:
                                        self_con=f[pathprefix+closekey+pathsuffix].attrs['self_consistency']
                                        if isinstance(self_con, float):
                                                tolstr='tol_'+str(round(self_con, roundto))
                                        else:
                                                if isinstance(self_con[-1], float):
                                                        tolstr='tol_'+str(round(self_con[-1], roundto))
                                                else:
                                                        tolstr='tol_'+str(round(self_con[-1][0], roundto))
                                        tolstr=closest_key(tolstr, np.float64, f[pathprefix+closekey+pathsuffix], noself=False, smallereq=True)
                                        if tolstr not in f[pathprefix+closekey+pathsuffix]:
                                                tolstr=closest_key(tolstr, np.float64, f[pathprefix+closekey+pathsuffix])
                                        if tolstr in f[pathprefix+closekey+pathsuffix]:
                                                if 'delta_i' in f[pathprefix+closekey+pathsuffix+'/'+tolstr]:
                                                        delta_i_closekey=f[pathprefix+closekey+pathsuffix+'/'+tolstr]['delta_i'][:]
                                if 'self_consistency' in f[pathprefix+closekey+pathsuffix]:
                                        self_con=f[pathprefix+closekey+pathsuffix]['self_consistency'][:,:]
                                        tolstr='tol_'+str(round((self_con[np.nonzero(self_con[:,0]),0])[0,-1], roundto))

                                        tolstr=closest_key(tolstr, np.float64, f[pathprefix+closekey+pathsuffix], noself=False, smallereq=True)
                                        if tolstr not in f[pathprefix+closekey+pathsuffix]:
                                                tolstr=closest_key(tolstr, np.float64, f[pathprefix+closekey+pathsuffix])
                                        if tolstr in f[pathprefix+closekey+pathsuffix]:
                                                if 'delta_i' in f[pathprefix+closekey+pathsuffix+'/'+tolstr]:
                                                        delta_i_closekey=f[pathprefix+closekey+pathsuffix+'/'+tolstr]['delta_i'][:]
                        if delta_i_key is not None and delta_i_closekey is not None and np.mean(np.abs((delta_i_key-delta_i_closekey)/delta_i_closekey)) < tolerance:
                                check=True
                        if delta_i_key is None or delta_i_closekey is None:
                                check=True
                        return check

                roundto=6
                self_con=None
                tolstr=None
                iterations=None
                delta_i=None
                n_i=None
## change to None as soon as all files include mu!
                mu=0.
## change to None as soon as all files include mu!
                cycle_time=0.
                n_=round(n, roundto)
                T_=round(T,2*roundto)
                W_=round(W,roundto)
                U_=round(U,roundto)
                nstr='n_'+str(str(n_))
                Tstr='T_'+str(T_)
                Wstr='W_'+str(W_)
                Ustr='U_'+str(U_)
                Nmstr='Nm_'+str(Nm)
#       implement in a better way, when time (deleteNm)
                deleteNm=False
#       implement in a better way, when time (deleteNm)
                if 'n' in mode:
                        nstr=closest_key(nstr, np.float64, f)
                if nstr in f:
                        if 'T' in mode:
#                               Tstr=closest_key(Tstr, np.float64, f[nstr])
                                Tstr=closest_key(Tstr, np.float64, f[nstr], smallereq=True)
                        if Tstr in f[nstr]:
                                if 'W' in mode:
                                        Wstr=closest_key(Wstr, np.float64, f[nstr][Tstr])
                                if Wstr in f[nstr][Tstr]:
                                        if 'U' in mode:
                                                Ustr=closest_key(Ustr, np.float64, f[nstr][Tstr][Wstr])
                                        if Ustr in f[nstr][Tstr][Wstr]:
                                                if 'Nm' in mode:
#                                                       newTstr=closest_key(Tstr, np.float64, f[nstr])
#                                                       newWstr=closest_key(Wstr, np.float64, f[nstr][Tstr])
                                                        newUstr=closest_key(Ustr, np.float64, f[nstr][Tstr][Wstr])
                                                        if ('c' not in mode):
                                                                Nmstr=closest_key(Nmstr, np.int, f[nstr][Tstr][Wstr][Ustr])
#                                                       elif checkkey(Nmstr, nstr+'/'+newTstr+'/'+Wstr+'/'+Ustr+'/', '', np.int, f): 
#                                                       elif checkkey(Nmstr, nstr+'/'+Tstr+'/'+newWstr+'/'+Ustr+'/', '', np.int, f): 
                                                        elif checkkey(Nmstr, nstr+'/'+Tstr+'/'+Wstr+'/'+newUstr+'/', '', np.int, f): 
#                                                               Nmstr=closest_key(Nmstr, np.int, f[nstr][newTstr][Wstr][Ustr], noself=True, smallereq=True)
#                                                               Nmstr=closest_key(Nmstr, np.int, f[nstr][Tstr][newWstr][Ustr], noself=True, smallereq=True)
                                                                Nmstr=closest_key(Nmstr, np.int, f[nstr][Tstr][Wstr][newUstr], noself=True, smallereq=True)
                                                        else:
                                                                Nmstr=''
                                                                deleteNm=True
                                                if Nmstr in f[nstr][Tstr][Wstr][Ustr]:
                                                        if 'time' in f[nstr][Tstr][Wstr][Ustr][Nmstr].attrs:
                                                                cycle_time=f[nstr][Tstr][Wstr][Ustr][Nmstr].attrs['time']
                                                                if not isinstance(cycle_time, float):
                                                                        cycle_time=sum(cycle_time)
                                                        if 'self_consistency' in f[nstr][Tstr][Wstr][Ustr][Nmstr].attrs:
                                                                self_con=f[nstr][Tstr][Wstr][Ustr][Nmstr].attrs['self_consistency']
                                                                if isinstance(self_con, float):
                                                                        tolstr='tol_'+str(round(self_con, roundto))
                                                                else:
                                                                        if isinstance(self_con[-1], list) or isinstance(self_con[-1], np.ndarray):
                                                                                tolstr='tol_'+str(round(self_con[-1][0], roundto))
                                                                        else:
                                                                                tolstr='tol_'+str(round(self_con[-1], roundto))
                                                                tolstr=closest_key(tolstr, np.float64, f[nstr][Tstr][Wstr][Ustr][Nmstr], noself=False, smallereq=True)
                                                                if tolstr not in f[nstr][Tstr][Wstr][Ustr][Nmstr]:
                                                                        tolstr=closest_key(tolstr, np.float64, f[nstr][Tstr][Wstr][Ustr][Nmstr])
                                                                if tolstr in f[nstr][Tstr][Wstr][Ustr][Nmstr]:
                                                                        if 'iterations' in f[nstr][Tstr][Wstr][Ustr][Nmstr][tolstr].attrs:
                                                                                iterations=np.int(f[nstr][Tstr][Wstr][Ustr][Nmstr][tolstr].attrs['iterations'])
                                                                        if 'delta_i' in f[nstr][Tstr][Wstr][Ustr][Nmstr][tolstr]:
                                                                                delta_i=f[nstr][Tstr][Wstr][Ustr][Nmstr][tolstr]['delta_i'][:]
                                                                        if 'n_i' in f[nstr][Tstr][Wstr][Ustr][Nmstr][tolstr]:
                                                                                n_i=f[nstr][Tstr][Wstr][Ustr][Nmstr][tolstr]['n_i'][:]
                                                                        if 'mu' in f[nstr][Tstr][Wstr][Ustr][Nmstr][tolstr].attrs:
                                                                                mu=dtype(f[nstr][Tstr][Wstr][Ustr][Nmstr][tolstr].attrs['mu'])
                                                        if 'self_consistency' in f[nstr][Tstr][Wstr][Ustr][Nmstr]:
                                                                self_con=f[nstr][Tstr][Wstr][Ustr][Nmstr]['self_consistency'][:,:]
                                                                tolstr='tol_'+str(round((self_con[np.nonzero(self_con[:,0]),0])[0,-1], roundto))
                                                                
                                                                tolstr=closest_key(tolstr, np.float64, f[nstr][Tstr][Wstr][Ustr][Nmstr], noself=False, smallereq=True)
                                                                if tolstr not in f[nstr][Tstr][Wstr][Ustr][Nmstr]:
                                                                        tolstr=closest_key(tolstr, np.float64, f[nstr][Tstr][Wstr][Ustr][Nmstr])
                                                                if tolstr in f[nstr][Tstr][Wstr][Ustr][Nmstr]:
                                                                        if 'iterations' in f[nstr][Tstr][Wstr][Ustr][Nmstr][tolstr].attrs:
                                                                                iterations=np.int(f[nstr][Tstr][Wstr][Ustr][Nmstr][tolstr].attrs['iterations'])
                                                                        if 'delta_i' in f[nstr][Tstr][Wstr][Ustr][Nmstr][tolstr]:
                                                                                delta_i=f[nstr][Tstr][Wstr][Ustr][Nmstr][tolstr]['delta_i'][:]
                                                                        if 'n_i' in f[nstr][Tstr][Wstr][Ustr][Nmstr][tolstr]:
                                                                                n_i=f[nstr][Tstr][Wstr][Ustr][Nmstr][tolstr]['n_i'][:]
                                                                        if 'mu' in f[nstr][Tstr][Wstr][Ustr][Nmstr][tolstr].attrs:
                                                                                mu=dtype(f[nstr][Tstr][Wstr][Ustr][Nmstr][tolstr].attrs['mu'])
                if mode!='':
                        iterations=0
                        self_con=[[np.float64('Inf'), np.float64('Inf'), np.float64('Inf')]]
                        cycle_time=0.
#       not valid for n!=0.875
#                       if 'U' in mode:
#                               Uold=np.float64(Ustr.split('_')[1])
#                               if delta_i is not None:
#                                       delta_i=delta_i*np.exp(-5.*(Uold-U_)/(Uold*U_))
#       not valid for n!=0.875
                return delta_i, n_i, mu, iterations, self_con, cycle_time, deleteNm

        seed=None
        n_i=None
        delta_i=None
        iterstart=0
        self_con=None
        mu=mustart
        cycle_time=0.
        deleteNm_=False
        try:
                with h5py.File(path, 'r') as f:
                        if 'seed' in f.attrs:
                                seed=f.attrs['seed']
                        for m in mode:
                                datalist=get_data(f, mode=m)
                                if None not in datalist:
                                        delta_i, n_i, mu, iterstart, self_con, cycle_time, deleteNm_=datalist
                                        break 
                
        except IOError:
                sys.exc_clear() 

        if not deleteNm:
                deleteNm_=False
        if n_i is None:
                n_i=np.ones(L*L, dtype=dtype)*n
        if delta_i is not None and np.mean(delta_i)<(1e-16/tol):
                delta_i=np.ones(L*L, dtype=dtype)*100*(1e-16/tol)
        if delta_i is None:
                delta_i=np.ones(L*L, dtype=dtype)*delta_start
        if self_con is None:
                self_con=np.array([[np.float64('Inf'), np.float64('Inf'), np.float64('Inf')]])
        if mu is None:
                mu=mustart
        return delta_i, n_i, mu, iterstart, self_con, cycle_time, seed, deleteNm_

def rescaleArray(ham, delta_i, onsite, eps=0.1):
        """
        This is a function to rescale the hams and eigenvalues
        Ham   : Input sparse matrix that is only get returned
        emin  : smallest eigen value
        emax  : largest eigen value
        eps   : small cuttoff to avoid stability issues
        """

        ndim = ham.shape[0]

        emax = np.real(arp.eigsh(ham, k=1,
                which='LA',return_eigenvectors=False,maxiter=100000, tol=5.0e-4))[0]
        emin=np.float64(-emax)

        a = (emax - emin)/(2.0-eps)
#       b = (emax + emin)/2.0

#       onsite=(onsite-b)/a
        onsite=onsite/a
        pairing=delta_i/a

        return onsite, pairing, np.float64(1./a), emax


def moment_evaluation(onsite_new, pairing_new, nDim, n, T, U, Nm, mu, hopping, dtype=np.float64):
        onsite_new_final=np.zeros(nDim//2)
        pairing_new_final=np.zeros(nDim//2)
#       onsite_new=onsite_new/(nRndVec*chebynRnd)
        onsite_new=np.reshape(onsite_new, (-1,Nm))
#       pairing_new=pairing_new/(nRndVec*chebynRnd)
        pairing_new=np.reshape(pairing_new, (-1,Nm))
        moments=np.arange(0,Nm)
        moments=((Nm-moments+1)*np.cos(np.pi*moments/(Nm+1))+np.sin(np.pi*moments/(Nm+1))/np.tan(np.pi/(Nm+1)))/(Nm+1)
        Npts=2*Nm
        dos=np.zeros(Npts, dtype=dtype)
        pairingdos=np.zeros(Npts, dtype=dtype)
        if T!=0.:
                fermivec=fermi(1./(T*hopping), np.cos(np.pi*(np.arange(Npts//2, Npts, dtype=dtype)+0.5)/Npts))
                for i in range(nDim//2): 
                        onsite_cosine=fftpack.dct(np.multiply(moments, onsite_new[i,:]),type=3,n=Npts)
                        onsite_new_final[i]=2.*np.sum(onsite_cosine[len(onsite_cosine)//2:]*fermivec+onsite_cosine[:len(onsite_cosine)//2]*(1.-fermivec[::-1]))/Npts
                        pairing_cosine=fftpack.dct(np.multiply(moments,pairing_new[i,:]),type=3, n=Npts)
                        pairing_new_final[i]=np.sum(pairing_cosine[len(pairing_cosine)//2:]*(1-2.*fermivec))/Npts
                        dos+=onsite_cosine
                        pairingdos+=pairing_cosine
        else:
                for i in range(nDim//2): 
                        onsite_cosine=fftpack.dct(np.multiply(moments, onsite_new[i,:]),type=3,n=Npts)
                        onsite_new_final[i]=2.*np.sum(onsite_cosine[len(onsite_cosine)//2:])/Npts
                        pairing_cosine=fftpack.dct(np.multiply(moments,pairing_new[i,:]),type=3, n=Npts)
                        pairing_new_final[i]=-1.*np.sum(pairing_cosine[len(pairing_cosine)//2:])/Npts
                        dos+=onsite_cosine
                        pairingdos+=pairing_cosine

        xarr=np.cos(np.pi*(np.arange(0.,2*Nm, 1.)+0.5)/(2*Nm))
        dos=dos[::-1]/np.sqrt(1.-xarr[::-1]**2)/(0.5*np.pi*(nDim//2))
        pairingdos=pairingdos[::-1]/np.sqrt(1.-xarr[::-1]**2)/(np.pi/U*(nDim//2))
        
        return U*pairing_new_final, onsite_new_final, mu-(np.mean(onsite_new_final)-n)*U/2., dos, pairingdos

    
def self_consistent_potentials(delta_i_old, n_i_old, nRndVec, nDim, n, T, U, Nm, pairing, onsite, mu, hopping, dtype=np.float64):#, mixing=1.0):
        """
        This fucntion calculates the LDOS and then return an array
        For non-interacting electrons

        nDim : dimension of the problem
        Nm   : # moment
        matH : Hamiltonian
        xx   : Points to be calculated
        """
        onsite_new=np.zeros(nDim//2*Nm, dtype=dtype)
        pairing_new=np.zeros(nDim//2*Nm, dtype=dtype)
        rndvec=np.zeros(nDim*chebynRnd, dtype=dtype)
        rndvec_nonzero=0
        printstep=5
        for i in range(nRndVec):
                rndvec[:]=(np.random.rand(nDim*chebynRnd)-0.5)*np.sqrt(12.)
                cy_chebyshev_cython_double.mychebyshev(onsite, pairing, rndvec, rndvec_nonzero, Nm, hopping, onsite_new, pairing_new)
                if (i+1)%printstep==0:
                        print('nRndVec', chebynRnd*(i+1))
                        pairing_new_final, onsite_new_final, mu, dos, pairingdos=moment_evaluation(np.array(onsite_new/np.float64((i+1)*chebynRnd)), np.array(pairing_new/np.float64((i+1)*chebynRnd)), nDim, n, T, U, Nm, mu, hopping, dtype=dtype)
#                       pairing_new_final, onsite_new_final, mu, dos, pairingdos=moment_evaluation(np.array(onsite_new), np.array(pairing_new), nDim, n, T, U, Nm, mu, hopping, dtype=dtype)
#                       onsite_new_final=1.-onsite_new_final
                        print('max rel abs pairing', np.max(np.abs(1.-pairing_new_final/delta_i_old)))
                        print('mean rel abs pairing', np.mean(np.abs(1.-pairing_new_final/delta_i_old)))
                        print('mean rel pairing', np.mean(1.-pairing_new_final/delta_i_old))
                        print('mean diff pairing', np.mean(pairing_new_final-delta_i_old))
                        print('mean pairing', np.mean(pairing_new_final))
                        print('max rel abs onsite', np.max(np.abs(1.-onsite_new_final/n_i_old)))
                        print('mean rel abs onsite', np.mean(np.abs(1.-onsite_new_final/n_i_old)))
                        print('mean rel onsite', np.mean(1.-onsite_new_final/n_i_old))
                        print('mean diff onsite', np.mean(onsite_new_final-n_i_old))
                        print('mean onsite', np.mean(onsite_new_final))
                        print(pairing_new_final, onsite_new_final)

        pairing_new_final, onsite_new_final, mu, dos, pairingdos=moment_evaluation(onsite_new/(nRndVec*chebynRnd), pairing_new/(nRndVec*chebynRnd), nDim, n, T, U, Nm, mu, hopping, dtype=dtype)
        
        return pairing_new_final, onsite_new_final, mu, dos, pairingdos

def self_consistent_potentials_diag(nDim, n, T, U, Nm, ham, L, mu, dtype_=np.float64):
    """
    This fucntion calculates the LDOS and then return an array
    For non-interacting electrons

    nDim : dimension of the problem
    Nm   : # moment
    matH : Hamiltonian
    xx   : Points to be calculated
    """
    onsite_new=np.zeros(nDim//2, dtype=dtype_)
    pairing_new=np.zeros(nDim//2, dtype=dtype_)
    pairingdos=np.zeros(nDim//2, dtype=dtype_)
    dos=np.zeros(nDim, dtype=dtype_)

    e, ev = np.linalg.eigh(ham.toarray())
    e=e[L*L:]
    ev=ev[:, L*L:]            # use only eigenvectors with positive eigenvalue
    
    if T!=0.:
            fermivec=fermi(1./T, e)
            for i in range(L):
                for j in range(L):
                    pairingdos_tmp=ev[2*(i+j*L), :]*ev[2*(i+j*L)+1, :]
                    pairingdos+=pairingdos_tmp
                    pairing_new[i+j*L]=U*np.sum((1.-2.*fermivec)*pairingdos_tmp)  # construct delta
                    u_tmp=ev[2*(i+L*j), :]*ev[2*(i+L*j), :]
                    v_tmp=ev[2*(i+L*j)+1, :]*ev[2*(i+L*j)+1, :]
                    dos[0:nDim//2]+=v_tmp
                    dos[nDim//2:]+=u_tmp
                    onsite_new[i+j*L]=2.*(np.sum(v_tmp*(1.-fermivec))+np.sum(u_tmp*fermivec)) #construct mu_i_new
            dos/=(nDim//2)
            pairingdos*=U
    else:
            for i in range(L):
                for j in range(L):
                    pairingdos_tmp=ev[2*(i+j*L), :]*ev[2*(i+j*L)+1, :]
                    pairingdos+=pairingdos_tmp
                    pairing_new[i+j*L]=U*np.sum(pairingdos_tmp)  # construct delta
                    u_tmp=ev[2*(i+L*j), :]*ev[2*(i+L*j), :]
                    v_tmp=ev[2*(i+L*j)+1, :]*ev[2*(i+L*j)+1, :]
                    dos[0:nDim//2]+=v_tmp
                    dos[nDim//2:]+=u_tmp
                    onsite_new[i+j*L]=2.*np.sum(v_tmp) #construct mu_i_new
            dos/=(nDim//2)
            pairingdos*=U

    mu_new=mu-(np.mean(onsite_new)-n)*U/2.
 
    return  pairing_new, onsite_new, mu_new, dos, pairingdos, e

def BdGHam(N, t, Wc, U, delta, n_i, mu, d=2, periodic=True,  seed=None, dtype=np.float64):
        """
        This will calculate the current operator for lattice
        models. It will return a sparse array.
        N : linear dimension of the problem.
        d : dimension of the system(1,2,3)
        t : hopping amplitude
        """
        # disorder
        if seed is None:
                seed=SystemRandom().getrandbits(32)    # set seed
        RndHam_Gen=np.random.RandomState(seed)

        n_i=n_i.astype(dtype)
        delta=delta.astype(dtype)
        Vr=(Wc*(RndHam_Gen.rand(N**d)-0.5)-U/2.*n_i-mu).astype(dtype)                    # Onsite Energy

        pairing=sps.kron(sps.diags(delta, 0), sps.lil_matrix(np.array([[0, 1], [0, 0]], dtype=dtype)).tocsr())+sps.kron(sps.diags(np.conjugate(delta), 0), sps.lil_matrix(np.array([[0, 0], [1, 0]], dtype=dtype)).tocsr())          # Pairing Term in Hamilton as Sparse Matrix

        onsite=sps.kron(sps.diags(Vr, 0), sps.lil_matrix(np.array([[1, 0], [0, 0]], dtype=dtype)).tocsr())+sps.kron(sps.diags(-1*np.conjugate(Vr), 0), sps.lil_matrix(np.array([[0, 0], [0, 1]], dtype=dtype)).tocsr())                         # Onsite Energy as Sparse Matrix

        nn = np.ones(N-1, dtype=dtype)
        #always returns sparse matrix

        hopm = -t*sps.lil_matrix(np.array([[1, 0], [0, 0]], dtype=dtype)).tocsr()+np.conjugate(t)*sps.lil_matrix(np.array([[0, 0], [0, 1]], dtype=dtype)).tocsr()

        hop = sps.kron(sps.diags(nn,1, dtype=dtype), hopm) + sps.kron(sps.diags(nn,-1, dtype=dtype), np.conjugate(hopm))
          
        if periodic==True:
                bc = sps.kron(sps.diags([1.],N-1, dtype=dtype), hopm) + sps.kron(sps.diags([1.],-(N-1), dtype=dtype), np.conjugate(hopm))
                h1d = hop + bc
        if d==1:
                ham  = h1d
        if d>1:
                ham  = sps.kron(sps.identity(N, dtype=dtype),h1d, format="csr")
                hop  = sps.kron(sps.kron(sps.diags(np.ones(N-1, dtype=dtype),1, dtype=dtype), np.identity(N, dtype=dtype)),  hopm)
                bc  = sps.kron(sps.kron(sps.diags([1.], N-1, dtype=dtype), np.identity(N, dtype=dtype)),  hopm)
                
                ham  = ham + hop + hop.H + bc + bc.H
        if d>2:
                ham  = sps.kron(sps.identity(N, dtype=dtype),ham, format="csr")
                
                hop = sps.kron(sps.kron(sps.kron(sps.diags(np.ones(N-1, dtype=dtype),1, dtype=dtype), np.identity(N, dtype=dtype)), np.identity(N, dtype=dtype)), hopm)
                bc  = sps.kron(sps.kron(sps.kron(sps.diags([1.], N-1, dtype=dtype), np.identity(N, dtype=dtype)), np.identity(N, dtype=dtype)), hopm)
                            
                ham  = ham + hop + hop.H + bc + bc.H
                
        ham = ham + onsite + pairing

        return ham, Vr

def testselfconsistency(delta, delta_old, n_i,  n_i_old, mu, mu_old, atol=1e-8, tol=0.01, minimum_mean_delta=1e-3):           # test if self-consistency was achieved
        self_consistency=[]
        # delta_i
        if np.mean(delta)<(minimum_mean_delta):
                self_consistency.append(0.)
        else:
                diffarr=np.abs(delta-delta_old)
                atoli=np.where(diffarr>atol)
                maxi=np.argmax(diffarr[atoli]/np.abs(delta[atoli]))
                self_consistency.append(diffarr[maxi]/delta[maxi])
        # n_i
        diffarr=np.abs(n_i-n_i_old)
        atoli=np.where(diffarr>atol)
        maxi=np.argmax(diffarr[atoli]/np.abs(n_i[atoli]))
        self_consistency.append(diffarr[maxi]/n_i[maxi])
        # mu
        self_consistency.append((mu-mu_old)/mu)
        return np.array(self_consistency)

def self_consistency_loop(nRndVec, L=16, dim=2, n=0.875, T=0., Nm=8192,  W=0.5, U=2.0, niter=10000, seed=None, path=None, tolerance=0.05, dtype=np.float64, initial_time=0, delta_i=np.ones(256**2)*0.1, n_i=np.ones(256**2)*0.875, mu=-0.1, iterstart=0, self_consistency=[np.float64('Inf'), np.float64('Inf'), np.float64('Inf')], cycle_time=0., diag=False):

        ndim=L**dim             # number of sites
        t=np.float64(1.0)

#       mixing=1.0
#       mixing_threshold=0.0
#       mixing_after=1.0

        tolcheckmax=0.1
        tolcheckstep=0.05
        roundto=6

        tolcheckarr=np.arange(tolcheckmax, tolerance-0.01*tolerance, -tolcheckstep)
        while tolerance<tolcheckmax:
                tolcheckmax=tolcheckmax*0.1
                tolcheckstep=tolcheckstep*0.1
                tolcheckarr=np.array(list(tolcheckarr)+list(np.arange(tolcheckmax, tolerance-0.01*tolerance, -tolcheckstep)))
        tolcheckarr=tolcheckarr[np.where([any(tolel<np.abs(self_con) for self_con in self_consistency[iterstart,:]) for tolel in tolcheckarr])]
        if tolerance not in tolcheckarr:
                tolcheckarr=np.array(list(tolcheckarr)+[tolerance])

        if seed is None:
                seed=cryptogen.getrandbits(32)    # set seed

        initialize_file(path, nRndVec, n, T, W, U, Nm, seed, delta_i, n_i, iterstart, dtype)

        for j in range(iterstart,niter):
                starttime=time.time()
                ##1 create HAM first
                ham, onsite = BdGHam(L, t, W, U, delta_i, n_i, mu,  d=dim,  seed=seed, dtype=dtype)

                ##2 rescale stuff
                if not diag:
                        onsite, pairing, hopping, emax = rescaleArray(ham, delta_i, onsite)
                ##3 determine the self-consistent potentials
                n_i_old=n_i
                delta_i_old=delta_i
                mu_old=mu
                print('delta_i', delta_i)
                print('n_i', n_i)
                print('mean delta_i', np.mean(delta_i))
                print('mean n_i', np.mean(n_i))
                if diag:
                        delta_i, n_i, mu, dos, pairingdos, e=self_consistent_potentials_diag(2*ndim, n, T, U, Nm, ham, L, mu, dtype_=dtype)
                        emax=e[-1]
                else:
                        delta_i, n_i, mu, dos, pairingdos=self_consistent_potentials(delta_i_old, n_i_old, nRndVec, 2*ndim, n, T, U, Nm, pairing, onsite, mu_old, hopping, dtype=dtype)
                        e=False
                print(delta_i, n_i)
                print('max delta', np.max(np.abs(1.-delta_i/delta_i_old)))
                print('mean delta', np.mean(np.abs(1.-delta_i/delta_i_old)))
                print('max n', np.max(np.abs(1.-n_i/n_i_old)))
                print('mean n', np.mean(np.abs(1.-n_i/n_i_old)))

                endingtime=time.time()
                cycle_time+=np.float64(endingtime-starttime)

                maxrss= np.float64(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000)

                now=time.time()

                self_consistency[j+1,:]=testselfconsistency(delta_i,  delta_i_old, n_i,  n_i_old, mu, mu_old, atol=0., tol=tolerance)
                indarr=np.where([all(tolel>abs(self_con) for self_con in self_consistency[j+1,:]) for tolel in tolcheckarr])
                tolchecktmp=tolcheckarr[indarr]
#############################################
#               if tolchecktmp.size!=0:
#                       tolcheck=np.min(tolchecktmp)
#                       write_data(nRndVec,path, ndim, n, T, W, U, Nm, tolcheck, niter, delta_i, n_i, mu, j+1, cycle_time, self_consistency, maxrss, dos, pairingdos, emax, e, dtype=np.float64)
#               else:
#                       write_data(nRndVec,path, ndim, n, T, W, U, Nm, tolcheckarr[-1], niter, delta_i, n_i, mu, j+1, cycle_time, self_consistency, maxrss, dos, pairingdos, emax, e, dtype=np.float64)
#               tolcheckarr=np.delete(tolcheckarr, indarr)
#
#               if all(tolerance>abs(self_con) for self_con in self_consistency[j+1,:]):
#                       if (now-initial_time)/3600.+(1.1*cycle_time/(j+1))/3600.>=timelimit:
#                               return 1
#                       else: 
#                               return 0
#
#               if (now-initial_time)/3600.+(1.1*cycle_time/(j+1))/3600.>=timelimit:
#                       return 1
#########################################
                write_data(nRndVec,path, ndim, n, T, W, U, Nm, tolerance, niter, delta_i, n_i, mu, j+1, cycle_time, self_consistency, maxrss, dos, pairingdos, emax, e, dtype=np.float64)


        return    

if __name__=="__main__":
        initial_time=time.time()

        dtype=np.float64

        n=np.float64(0.3)

        Warr=[700]

        Uarr=[220, 180, 150]

        Tmin=0.0
        Tmax=0.08
        Tstep=9

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

        Nmarr={}
        
        for U in Uarr:
                U=dtype(round(0.01*U,2))
                Nmarr[U]={}
                if U!=1.5:
                        for W in Warr:
                                W=dtype(round(0.01*W,2))
                                Nmarr[U][W]={}
                                for iT in range(1):
                                        Nmarr[U][W][round(Tarr[W][U][iT],3)]=[1024, 2048]
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

        tolerance=0.00009

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
        nconfigs=5

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
                for nRndVec in [100]:
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
                                                        endflag=self_consistency_loop(nRndVec, L=L, n=n, T=T, Nm=Nm, W=W, U=U, niter=niter, path=directory+'/'+name, tolerance=tolerance, dtype=dtype, seed=seed, initial_time=initial_time, delta_i=delta_i, n_i=n_i, mu=mu, iterstart=iterstart, self_consistency=self_consistency, cycle_time=cycle_time, diag=diag)
                                                        if endflag==1:
                                                                break
