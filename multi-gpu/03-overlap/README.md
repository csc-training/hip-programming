<!-- 
SPDX-FileCopyrightText: 2025 CSC - IT Center for Science Ltd. <www.csc.fi> 
 
SPDX-License-Identifier: CC-BY-4.0 
--> 

# Overlapping communication and computation

Note: this exercise assumes a knowledge about basic MPI programming

In this exercise you can investigate how to hide communication costs
by overlapping communication with computation. As computations are done
in GPU asynchronously to the host CPU, CPU can participate in the message
progress and there is more potential for overlap than when using only CPUs.

The code [jacobi.cpp](jacobi.cpp) solves two dimensional Poisson equation with
Jacobi iteration (see end of this document for information about the method).
The code is parallelized with MPI in such a way that the two dimensional grid
is divided into block of rows, and each MPI tasks processes one block. At each
iteration step one first communicates the outermost rows and performs then 
computation.

## Tasks

1. Build the code
```
CC -x hip -o jacobi jacobi.cpp -O3 -lroctx64
```
and run it with 1, 2, 4 MPI tasks / GPUs. How does the performance improve when
increasing number of GPUs?

2. Run the code under `rocprofv3` with 4 MPI tasks / GPUs, e.g.
```
srun ... rocprofv3 -r --output-format pftrace ./jacobi
```
and investigate the trace. The communication step is traced with roctx markers.
(Note: https://github.com/IBM/mpitrace provides a simple mechanism for tracing 
all MPI calls with roctx markers).

3. When computing a single grid point, information only about the neighbouring grid points is needed.
Thus, it is possible to compute the inner part of the grid separately from the outermost rows,
and perform the communication concurrently to the computation of the inner region. Try to implement
this overlapping. Does the performance improve? How does the trace look now?

## Background about Poisson equation and Jacobi iteration

Poisson's equation in 2D is

$$
\nabla^2 u(x, y) = f(x, y)
$$

where $$f$$ is a given function and $$u$$ is the function to be solved.

This equation can be discretized using finite differences as:

$$
\frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{\Delta x^2} + \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{\Delta y^2} = f_{i,j}
$$

Assuming a uniform grid where $$\Delta x = \Delta y = h$$, this simplifies to:

$$
\frac{u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}}{h^2} = f_{i,j}
$$

Rearranging terms gives the standard five-point stencil:

$$
u_{i,j} = \frac{1}{4} \left( u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - h^2 f_{i,j} \right)
$$

This discretized equation can be solved iteratively using various numerical methods, and in this exercise,
we solve it using Jacobi iteration that updates all grid points simultaneously using values from the previous iteration.

The algorithm uses the five-point stencil:

$$
u_{i,j}^{(k+1)} = \frac{1}{4} \left( u_{i+1,j}^{(k)} + u_{i-1,j}^{(k)} + u_{i,j+1}^{(k)} + u_{i,j-1}^{(k)} - h^2 f_{i,j} \right)
$$

where $$u_{i,j}^{(k)}$$ is the value of $$u$$ at grid point $$(i,j)$$ during the $$k$$-th iteration.

The problem can be solved in parallel with MPI using domain decomposition: the grid is divided into blocks of rows
and single MPI task is assigned to each block. The stencil update requires information from the neighbouring  grid points,
so at each iteration step a halo exchange is performed. Grid accommodates empty "ghost" rows at the both boundaries,
and these ghost rows are updated by the real data from the two neighbouring MPI tasks / domains.

The algorithm comprises of the following steps:

1. Initialize the grid with an initial guess $$u_{i,j}^{(0)}$$ (zeros in the example code).
2. Iterate over all interior grid points and update $$u_{i,j}^{(k+1)}$$ using values from $$u^{(k)}$$.
3. Perform halo exchange, i.e. communicate outermost rows between MPI tasks
4. Repeat until the solution converges, i.e., the difference between successive iterations is below a chosen tolerance.
   In this exercise, we iterate for a fixed number of iterations for simplicity.



