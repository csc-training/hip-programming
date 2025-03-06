# [Discrete Hankel transform](https://en.wikipedia.org/wiki/Hankel_transform#Fourier_transform_in_two_dimensions) for CUDA enabled gpu.

Example of how to do CUDA quasi-discret Hankel transform. The space grid is obtained from the solutions of the first kind Bessel functions j0(x)=0. The solutions must be provided in advanced in the file bessel_zeros.in. The same function is used for for doing forward or backward transform. If a forward `(r->k)` transform is made the results must be multiplied by `rmax/kmax`. All the arrays needed for transform are kept in a structure which is iniliazed by the hankelinit function. Only double precision is available. The flag -arch=sm_xx should have xx>=20 .

Effectvely this is a matrix-vector multiplication, `C=H*A`. Because lots of points are needed, the matrix [`H`](https://openresearchsoftware.metajnl.com/articles/10.5334/jors.82/) exceeds the memory capacity of the gpu and the matrix is constructed on the fly using the Bessel function `j0`. The shared memory is used as a user-controled cache to avoid recomputing the local part of the matrix `H`or extra memory reads. The number of threads per block `tpb` can be changed for checking the performance. 
Compile with:
```nvcc  -archm=sm_75 Code.cu```

Lots of thanks to njuffa for giving very good advice on devtalk nvidia forum.
