# self_consistency_solver

The Python modules in the folder /Modules/ contain different versions of a self-consistency solver all containing a full diagonalization (for number of moments Nm=0) and Kernel Polynomial Method solver (for Nm>0). The full diagonalization solver is the same in every module.

1. The KPM solver of "self_consistency_MatrixFree_bdg.py" contains an optimized matrix-free sparse matrix product (written in C) that only works with TR- and PH- symmetric s-wave superconductors as outlined in our publication.
2. The KPM solver of "self_consistency_MKL_bdg.py" contains a MKL sparse matrix product (python wrapper written by Dominik Hahn) that can be used for any Hamiltonian. To use this solver for a different system than the implemented one, the self-consistency equations, Hamiltonian and details of the polynomial expansion have to be adjusted.
3. The KPM solver of "self_consistency_AllInPython_bdg.py" is written entirely in Python but very slow for that reason.
4. The KPM solver of "self_consistency_RandomTraceEvaluation_bdg.py" uses a Random Trace Evaluation approach but needs optimization to be practical.

Installation (only necessary for 1. and 2.):
1. In the folder "/chebyshev_matrixfree" in the file "chebyshev_final_icc_double.c" determine the linear dimension of the system with the variable N and the number threads with the variable NrndVec. Then execute make in folder "/chebyshev_matrixfree".
2. In the folder "/chebyshev_matrix" execute make.

Example job scripts are given with "start_self_consistency_solver.py" and "start_self_consistency_solver_MKL.py" for codes 1. and 2. respectively.

Start the script with:
python start_self_consistency_solver.py folder startfile system_size

In "folder" the self-consistent-fields will be written and read in every step. If the code has been stoppen before self-consistency was reached, it automatically restarts at the most recent iteration, if "folder" contains a file with the correct parameters and filename.
With "startfile" the first numbered suffix of the output filenames is defined. All the following output files will be numbered as defined in the script.
"system_size" defines the linear system size of the square lattice

All other parameters temperature, disorder strength, number of Chebyshev moments, particle density etc. are determined within the job scripts.
