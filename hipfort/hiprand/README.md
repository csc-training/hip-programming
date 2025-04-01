# Computing `pi`

HIPFORT provides interfaces for various highly optimized library. In this exercise `hip_rand` library is used to accelerate the computation of `pi` using Monte Carlo method. 

In this method, **random** `(x, y)` points are generated in a 2-D plane with domain as a square of side *2r* units centered on `(0,0)`.  
A circle of radius **r** centered at `(0,0)` will fit perfectly inside. The ratio between the area of circle and the square is `pi/4`. If a sufficiently large number of uniformly distributed random points are generated, the proportion of points inside the circle to the total points in the square provides a good estimate for `pi/4`. For efficiency, it is sufficient to consider only one-quarter of the domain  (i.e, where  `x>0` & `y>0`).

<figure>
  <img src="img/pi_MC.png" width="50%" alt="Pi Monte Carlo">
  <figcaption> </figcaption>
</figure>


## The task

In order to compute `pi` two arrays are filled with uniform distributed random numbers. Because it is (arguably) very expensive to compute random numbers on the CPU, this step is offloaded to GPU. 

In this exercise, one needs to complete the missing code sections labeled with **TODO** to obtain the correct result. This include adding the right Fortran modules to use, memory allocations, calling the `hiprand` library for generating random numbers, and transfering the data to CPU for the final counting. 

The memory allocations are done similarly to the [saxpy](../saxpy/hip) example. The random numbers are generate with `hiprand`. 

First one needs to first initialize the library:
```
istat= hiprandCreateGenerator(gen, HIPRAND_RNG_PSEUDO_DEFAULT)
```
Here, `istat` is an `integer(c_size_t)` variable and it is used to store the a value indicating if the call was succesful or failed, while `gen` is a pointer to the random number generator. In Fortran is declared as a `type(c_ptr)` variable. The generator type is set to `HIPRAND_RNG_PSEUDO_DEFAULT`, which is the default pseudorandom number generator.

A GPU array is filled with uniform distributed random numbers using:
```
istat= hiprandGenerateUniform(gen, A_d, n)
```
In this call the argument `A_d` is a `type(c_ptr)` variable in which the numbers are stored. It is assumed that the memory for it has been previously allocated. The last argument, `n` is the amount of random numbers to generate (should be the same as the size of the array). 

In the end there is one more step, counting the points inside the circle `(x^2+y^2<1)`. For this, first the arrays `x_d` and `y_d` are transfered from GPU to CPU. This is done as well similarly as in  the [saxpy](../saxpy/hip) example.

**Optional Bonus task** When the number of points `n` is large, counting points inside the circle on the CPU can become a bottleneck. You can offload this counting step to the GPU using a kernel that counts the points in parallel. 
In the last (optional, bonus) tasks one should offload to GPU the final loop:
```
    do i = 1, n
      if (x(i)**2 + y(i)**2 < 1.0) then
        inside = inside + 1
      end if
    end do
```
The most naive (a.k.a simple) implementation looks like this:
```
__global__ void countInsideCircle(float* x_d, float* y_d, int* inside_d,  int64_t n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        if (x_d[idx] * x_d[idx] + y_d[idx] * y_d[idx] < 1.0f) {
            atomicAdd(inside_d, 1);
        }
    }
}
```
The kernel checks each point in parallel and increments `inside_d` if the point lies within the circle. The `atomicAdd` function is used to avoid race conditions when updating the counter. In this code `atomicAdd` is called for each point inside circle, but an optimized version would have much less calls. 
In this case there is no need for transfering the arrays with the random numbers to the CPU. Only the result of the counting needs to be transferd to CPU. 
Note that an additional variable is needed to store the GPU result (could be `inside_d`). The actual kernel and its wrapper are found in the [hip_kernels.cpp](hip_kernels.cpp), but an interface is needed in the main fortran code to be able to call it. 

The Makefile needs as well to be updated to compile the C kernel along with the Fortran code. Similarly to `saxpy` example the fortran and C codes are compiled first to objects. So the list of objects will  now be 
```
OBJS=pi.o hip_kernels.o
```
Furthermore one has to add a rule  to create the `.o` file from the `.cpp` source:
```
%.o: %.cpp
	$(CXX) -c -o $@ $<
```
Finally there are other considerations. When doing calling C functions from Fortran the type of the variables should be compatible. Integers, single precision, and double preicision types are the same in both C and Fortran, but there are many others that are not. For ensuring full compatati bily and less debugging headaches in the future, it is important to use the `iso_c_binding` types. In this parrticular example `integer(kind=INT64) :: n` should be changed to `integer(c_int64_t) :: nsamples`.

For more examples of usage of HIPFORT and libraries please check also the [HIPFORT repository](https://github.com/ROCmSoftwarePlatform/hipfort/tree/develop/test).
