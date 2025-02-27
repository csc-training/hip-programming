# Computing `pi`

HIPFORT provides interfaces for various highly optimized library. In this exercise `hip_rand` library is used to accelerate the computation of `pi` using Monte Carlo method. 

In this method **random** `(x, y)` points are generated in a 2-D plane with domain as a square of side *2r* units centered on `(0,0)`.  
A circle of radius **r** centered at `(0,0)` will fit perfectly inside. The ratio between the area of circle and the square is `pi/4`. If a sufficiently large number of uniformly distributed random points are generated, the the proportion of points inside the circle to the total points in the square is a good estimate for `pi/4`. For efficiency, it is sufficient to consider only one-quarter of the domain  (i.e, where  `x>0` & `y>0`).

<figure>
  <img src="img/pi_MC.png" width="50%" alt="Pi Monte Carlo">
  <figcaption> </figcaption>
</figure>


## The task

In order to compute `pi` two arrays are filled with uniform distributed random numbers. Because it is (arguably) very expensive to compute random numbers, this step is offloaded to GPU. 

In this exercise, one needs to complete the missing code sections labeled with **TODO** to obtain the correct result. This include adding the right Fortran modules to use, memory allocations, calling the `hiprand` library for generating random numbers, and transfering the data to CPU for the final counting. 

The memory allocations are done similarly to the [saxpy](../saxpy/hip) example. While random numbers are generate with `hiprand`. 

First one needs to first initialize the library:
```
istat= hiprandCreateGenerator(gen, HIPRAND_RNG_PSEUDO_DEFAULT)
```
In the above `istat` is an `integer(c_size_t)` variable and it is used to store the a value indicating if the callwas succedful or failed. The arguments of the functions are `gen` a pointer to the random number generator, which in Fotran is declared as a `type(c_ptr)` variable and the type of generator which in this case is the default one. 

A GPU array is filled with uniform distribtued random numbers using:
```
istat= hiprandGenerateUniform(gen, A_d, n)
```
In this call the argument `A_d` is a `type(c_ptr)` variable and it is assumed that some memory was previously allocated to this variable. The last argument is an integer which indicates how many numbers are generated (shouldd be the same as the size of the array). 

In the end there is one more step, counting the points inside the circle `(x^2+y^2<1)`. For this first one needs to trasnfer the arrays `x_d` and `y_d`from GPU to CPU. This is done as well similarly as in  the [saxpy](../saxpy/hip) example.

**Optional task** When the size of the problem is very large a simple step like counting the points can take a large amount of time. 
In the last (optional) tasks one should offload to GPU the final loop:
```
    do i = 1, n
      if (x(i)**2 + y(i)**2 < 1.0) then
        inside = inside + 1
      end if
    end do
```
In this case there is no need for transfering the arrays with the random numbers to the CPU. Only the result of the counting needs to be transferd to CPU. 
Note that an additional variable is needed to store the GPU result (could be `inside_d.cpp`).
The actual kernel and wrapper are found in the [hip_kernels.cpp](hip_kernels.cpp), but an interface is needed in the main fortran code.

For more examples of usage of HIPFORT and libraries please check also the [HIPFORT repository](https://github.com/ROCmSoftwarePlatform/hipfort/tree/develop/test).
