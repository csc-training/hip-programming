# Peer to peer device access

Benchmark memory copies with and without peer to peer device access using two
GPUs.

Skeleton code [p2pcopy.cpp](p2pcopy.cpp) tests peer to peer device access between two GPUs by doing a series of memory copies. The test is evaluated after calling `hipDeviceEnablePeerAccess()` and `hipDeviceDisablePeerAccess()`. The program prints calculated bandwith and time for both cases. On a CUDA platform, there should be a difference in results, whereas on an AMD platform there is none. 

In order to make the code work, you need to fix the missing parts marked with TODOs.

NOTE: Remember to request 2 GPUs when running this exercise. On Lumi, use
```
srun --account=XXXXXX --partition=dev-g -N1 -n1 --cpus-per-task=1 --gpus-per-task=2 --time=00:15:00 ./a.out
```
and on Mahti use
```
srun --account=XXXXXX --partition=gputest -N1 -n1 --cpus-per-task=1 --gres=gpu:v100:2 --time=00:15:00 ./a.out
```
