#===============================================================================
# User Options
#===============================================================================
#
# Compiler can be set below, or via environment variable
CC        = nvcc
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
CFLAGS := -Xcompiler -Wall
# Linker Flags
LDFLAGS = -lcublas
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
	rm -rf $(program) $(obj)

