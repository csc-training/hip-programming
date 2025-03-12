# Converting CUDA code to HIP

The folder [codes](codes) contains a few examples (vector addition, `saxpy` using HIP  kernel, and `saxpy`using `cublas` of CUDA codes. On Mahti or Puhti these codes will compile with the CUDA `nvcc` compiler and should run without issues. 

The tasks are to convert these codes to HIP. For shorter code one can do a manual conversion, but for larger codes it is recomended to use HIPIFY tools or compile them with [HOP](https://github.com/cschpc/hop) library. 

## HIPIFY Tools
0. **Optional** Convert the codes to HIP manually. On Nvidia platforms the conversion can be done in an incremental way because `hipcc` can compile mixed CUDA and HIP code. On AMD plaftorms `hipcc` can not compile CUDA code. The whole code needs to be converted in order to be able to compile it. 
1. Convert the codes using HIPIFY tools.
   
    A. Examine the code. Both `hipify-perl` and `hipify-clang` support the option `--examine` option. Alternatively one can use the `hipexamine.sh`  or `hipexamine-perl.sh` scripts which will scan whole directories. This procedure will not change the source it will just determine which files contain CUDA code and how much of the code can be converted automatically.
   
    B. Convert individual files `hipify-[perl|clang] --inplace --print-stats` or folders using the scripts `hipconvertinplace[.|-perl.]sh <folder>`.


Note that on `hipify-clang` requires CUDA toolkit. On LUMI this is available via a container. 
The image can be created using:

```
$ singularity pull docker://nvcr.io/nvidia/cuda:11.4.3-devel-ubuntu20.04
```
Then load all the modules necessary to compile HIP codes on LUMI. 
```
module load LUMI/24.03
module load partition/G
module load rocm
```
Finally open a shell in the container which has access to the working directory and the `rocm` 
```
singularity shell -B $PWD,/opt:/opt cuda_11.4.0-devel-ubuntu20.04.sif 
export PATH=$ROCM_PATH/bin:$PATH
```

The CUDA code can be converted now  using:
```
hipify-clang <file>.cu --inplace --print stats  --cuda-path=/usr/local/cuda-11.4 -I /usr/local/cuda-11.4/include
```
This command works as well on Nvidia platforms with HIP installed. 

2. Compile CUDA codes on AMD platorms using `hipcc` + HOP and compile HIP codes on Nvidia platforms using `nvcc` + HOP.

**CUDA** &rArr; **HIP**
```
export HOP_ROOT=/path/to/hop
export HOP_FLAGS=-I$HOP_ROOT -I$HOP_ROOT/source/cuda -DHOP_TARGET_HIP
CC -x hip $HOP_FLAGS hello.cu -o hello
./hello
```
**HIP**  &rArr; **CUDA**
```
export HOP_ROOT=/path/to/hop
export HOP_FLAGS=-I$HOP_ROOT -I$HOP_ROOT/source/hip -DHOP_TARGET_CUDA
CC -x cu $HOP_FLAGS hello.cpp -o hello
./hello
```

