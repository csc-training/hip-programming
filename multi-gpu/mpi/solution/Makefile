HIPCC = hipcc
HIPCCFLAGS = -c -x=cu -O2
MPICXX = mpicxx
MPICXXFLAGS = -g -O2
LD = $(HIPCC)
LDFLAGS = -L/appl/spack/install-tree/gcc-9.1.0/hpcx-mpi-2.4.0-dnpuei/lib
LIBS = -lmpi
INCLUDES =
SRC = mpiexample.cpp hipwrapper.cpp
OBJS = $(SRC:.cpp=.cpp.o)
EXE = mpiexample


.SUFFIXES:


all: $(EXE)

$(EXE): $(OBJS)
	$(LD) $(LDFLAGS) -o $(EXE) $(OBJS) $(LIBS)

hipwrapper.cpp.o: hipwrapper.cpp
	$(HIPCC) $(HIPCCFLAGS) $(INCLUDES) -o $@ $<

mpiexample.cpp.o: mpiexample.cpp
	OMPI_CXXFLAGS='' OMPI_CXX='hipcc --x cu' $(MPICXX) $(MPICXXFLAGS) $(INCLUDES) -c -o $@ $<

.PHONY: clean

clean:
	rm -f $(OBJS) *~ $(EXE) *.o
