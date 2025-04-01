# Vector sum on two GPUs without MPI

Calculate the vector sum of two vectors (C = A + B) using two GPUs.

Decompose the vectors into equal halves, copy data from host to device memory
and launch a GPU kernel on each part asynchronously using streams. Copy the
results back to the host to check for correctness. Add timing events to
measure the time of execution.

A skeleton code is provided in [vector-sum.cpp](vector-sum.cpp). Your task is to fill the locations indicated by 

```// TODO:```

NOTE: Remember to request 2 GPUs when running this exercise. On Lumi, use
```
srun --account=XXXXXX --partition=small-g -N1 -n1 --cpus-per-task=1 --gpus-per-node=2 --time=00:15:00 ./a.out  # The reservation is for small-g partition
```
and on Mahti use
```
srun --account=XXXXXX --partition=gputest -N1 -n1 --cpus-per-task=1 --gres=gpu:v100:2 --time=00:15:00 ./a.out
```
