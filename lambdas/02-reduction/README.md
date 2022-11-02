# Reductions with host-device lambdas and CUB

The purpose of this exercise is to use host-device lambda functions and the CUB library to create an efficient reduction kernel. The location of the missing parts of the kernel code are indicated by #error. The CUB library documentation may be useful, particularly [this example](https://nvlabs.github.io/cub/classcub_1_1_block_reduce.html#a7632bd9c8950dd6a3528ca99fa3f0890).

IMPORTANT NOTE! When using the host-device lambda function with NVIDIA architecures, the following compiler argument must be added for hipcc: `--extended-lambda`
