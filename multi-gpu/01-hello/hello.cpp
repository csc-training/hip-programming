// SPDX-FileCopyrightText: 2026 CSC - IT Center for Science Ltd. <www.csc.fi>
//
// SPDX-License-Identifier: MIT

#include <mpi.h>
#include <hip/hip_runtime.h>
#include <cstdio>

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  // Will hold the actual length of the processor name (filled in by MPI)
  int processor_name_length;
  MPI_Get_processor_name(processor_name, &processor_name_length);

  int num_devices = 0; 
  int my_device = -1;

  int node_rank;
  // TODO obtain node local rank by creating "SHARED" communicator 
  MPI_Comm node_comm;

  // TODO query number of devices and set different device for each local MPI rank

  if (0 == rank) {
    printf("Total number of MPI processes: %d\n", size);
    printf("Number of GPUs per node: %d\n", num_devices);
  }

// Try to print in synchronized manner
  for (int i=0; i < size; i++) {
    if (rank == i) {
      printf("Global rank: %d in node %s Local rank: %d using GPU %d\n",
             rank, processor_name, node_rank, my_device);
      fflush(stdout);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Finalize();
  
  return 0;
}
