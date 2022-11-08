# Reductions with host-device lambdas and hipCUB

The purpose of this exercise is to use host-device lambda functions and the hipCUB library to create an efficient reduction kernel. The location of the missing parts of the kernel code are indicated by #error. The CUB library documentation may be useful, particularly [this example](https://nvlabs.github.io/cub/classcub_1_1_block_reduce.html#a7632bd9c8950dd6a3528ca99fa3f0890). Note that hipCUB uses namespace "hipcub" instead of "cub" used in the original CUDA library.

IMPORTANT NOTE! When using the host-device lambda function with NVIDIA architectures, the following compiler argument must be added for hipcc: `--extended-lambda`
