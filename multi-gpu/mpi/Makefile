HIPCC = hipcc
MPICXX = mpicxx
MPICXXENV = OMPI_CXXFLAGS='' OMPI_CXX='$(HIPCC) --x cu'
MPICXXFLAGS = -g -O2
LD = $(HIPCC)
LDFLAGS = -L/appl/spack/install-tree/gcc-9.1.0/openmpi-4.1.1-vonyow/lib
LIBS = -lmpi
INCLUDES =
SRC = mpiexample.cpp
OBJS = $(SRC:.cpp=.cpp.o)
EXE = mpiexample


.SUFFIXES:


all: $(EXE)

$(EXE): $(OBJS)
	$(LD) $(LDFLAGS) -o $(EXE) $(OBJS) $(LIBS)

mpiexample.cpp.o: mpiexample.cpp
	$(MPICXXENV) $(MPICXX) $(MPICXXFLAGS) $(INCLUDES) -c -o $@ $<

.PHONY: clean

clean:
	rm -f $(OBJS) *~ $(EXE) *.o
