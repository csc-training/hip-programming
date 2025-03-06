# Two dimensional heat equation

Example implementations of two dimensional heat equation with various parallel
programming approaches.

Heat (or diffusion) equation is 

$$\frac{\partial u}{\partial t} = \alpha \nabla^2 u $$

where **u(x, y, t)** is the temperature field that varies in space and time,
and α is thermal diffusivity constant. The two dimensional Laplacian can be
discretized with finite differences as

```math
\nabla^2 u  = \frac{u(i-1,j)-2u(i,j)+u(i+1,j)}{(\Delta x)^2} + \frac{u(i,j-1)-2u(i,j)+u(i,j+1)}{(\Delta y)^2}

```
 Given an initial condition (u(t=0) = u0) one can follow the time dependence
 of
 the temperature field with explicit time evolution method:

$$u^{m+1}(i,j) = u^m(i,j) + \Delta t \alpha \nabla^2 u^m(i,j) $$

 Note: Algorithm is stable only when

$$ \Delta t < \frac{1}{2 \alpha} \frac{(\Delta x \Delta y)^2}{(\Delta x)^2
 (\Delta y)^2} $$

## How to build

For building and running the example one needs to have the
[libpng](http://www.libpng.org/pub/png/libpng.html) library installed. Working MPI 
environment is required for all cases. In addition:

 * Hybrid MPI-OpenMP version requires MPI implementation with
   MPI_THREAD_MULTIPLE support
 * CUDA version requires CUDA environment and CUDA aware MPI

 Move to proper subfolder and modify the top of the **Makefile**
 according to your environment (proper compiler commands and compiler flags).
 Code can be build simple with **make**

## How to run

The number of MPI ranks has to be a factor of the grid dimension (default 
dimension is 2000). For GPU versions, number of MPI tasks per node has to be the
same as number of GPUs per node. 

The default initial temperature field is a disk. Initial
temperature field can be read also from a file, the provided **bottle.dat** 
illustrates what happens to a cold soda bottle in sauna.


 * Running with defaults: `srun <args> -N 4 ./heat_mpi`
 * Initial field from a file: `srun <args> -N 4 ./heat_mpi bottle.dat`
 * Initial field from a file, given number of time steps: `srun <args> -N 4 ./heat_mpi bottle.dat 1000`
 * Defauls pattern with given dimensions and time steps: `srun <args> -N 4 ./heat_mpi 800 800 1000` 

  The program produces a series of heat_XXXX.png files which show the
  time development of the temperature field

