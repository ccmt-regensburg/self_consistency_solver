CC = icc
PYTHON_INCLUDE="/home/stosiek/install/anaconda3/include/python3.8/"
PYTHON_LIB="/home/stosiek/install/anaconda3/lib/"

### Static link
LDFLAGS=  -shared -Wl,-rpath=${ORIGIN}/../lib/:${PYTHON_LIB} -L${PYTHON_LIB} -L${MKLROOT} -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lmkl_rt -lm -ldl -lmkl_avx2 -lmkl_def

#CFLAGS = -DNDEBUG -march=native -O3 -I$(MKLROOT)/include -Wall -fPIC -DMKL_ILP64
CFLAGS = -DNDEBUG -march=native -std=c99 -O3 -I${PYTHON_INCLUDE} -Wall -fPIC -DMKL_ILP64

.PHONY: all
all: cy_chebyshev_cython_parallel.so

sparse_d_mm_chebyshev_new.o: sparse_d_mm_chebyshev_new.c
		$(CC) $(CFLAGS) -fPIC -o sparse_d_mm_chebyshev_new.o -c $< 

libsparse_d_mm_chebyshev_new.so: sparse_d_mm_chebyshev_new.o
		$(CC) -shared  -o libsparse_d_mm_chebyshev_new.so $< $(LDFLAGS)

cy_chebyshev_cython_parallel.so:  libsparse_d_mm_chebyshev_new.so
		./cy_paralell.sh

		cp cy_chebyshev_cython_paralell.so ../Modules/
		cp libsparse_d_mm_chebyshev_new.so ../Modules/ 

.PHONY: clean
clean:
	rm -f libsparse_d_mm_chebyshev_new.so
	rm -f sparse_d_mm_chebyshev_new.o
	rm -f cy_chebyshev_cython_paralell.so
