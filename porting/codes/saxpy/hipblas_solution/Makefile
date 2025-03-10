#===============================================================================
# User Options
#===============================================================================
#
# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
#
#===============================================================================
# Program name & source code list
#===============================================================================
program = saxpy_cublas
source = saxpy_cublas.cu
obj = $(source:.cu=.o)
#===============================================================================
# Sets Flags
#===============================================================================
# Standard Flags
CFLAGS := -Xcompiler -Wall -I/appl/opt/rocm/rocm-4.0.0c/hipblas/hipblas/include
# Linker Flags
LDFLAGS = -L/appl/opt/rocm/rocm-4.0.0c/hipblas/hipblas/lib/ -lhipblas
# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

#===============================================================================
# Targets to Build
#===============================================================================
#
$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) out* error*

