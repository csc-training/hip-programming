ifeq ($(COMP),)
COMP=pgi
endif

COMMONDIR=../../common

ifeq ($(COMP),pgi)
FC=mpif90
CC=gcc
FCFLAGS=-O3 
CCFLAGS=-O3 -I$(COMMONDIR)
LDFLAGS=-Mcuda -L$(CUDA_INSTALL_ROOT)/lib64
LIBS=-lpng
endif

EXE=heat_cuda
OBJS=main.o heat_mod.o core.o core_cuda.o setup.o utilities.o io.o pngwriter_mod.o
OBJS_PNG=$(COMMONDIR)/pngwriter.o

all: $(EXE)

$(COMMONDIR)/pngwriter.o: $(COMMONDIR)/pngwriter.c $(COMMONDIR)/pngwriter.h
heat_mod.o: heat_mod.cuf
core.o: core.F90 heat_mod.o
core_cuda.o: core_cuda.cuf heat_mod.o
utilities.o: utilities.F90 heat_mod.o
io.o: io.F90 heat_mod.o pngwriter_mod.o 
setup.o: setup.F90 heat_mod.o utilities.o io.o
pngwriter_mod.o: pngwriter_mod.F90 heat_mod.o
main.o: main.F90 heat_mod.o core.o core_cuda.o io.o setup.o utilities.o

$(EXE): $(OBJS) $(OBJS_PNG)
	$(FC) $(FCFLAGS) $(OBJS) $(OBJS_PNG) -o $@ $(LDFLAGS) $(LIBS)

%.o: %.F90
	$(FC) $(FCFLAGS) -c $< -o $@

%.o: %.cuf
	$(FC) $(FCFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	-/bin/rm -f $(EXE) a.out *.o *.mod *.png *~
