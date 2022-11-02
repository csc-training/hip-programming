HIPCC = hipcc
MPICXX = mpicxx
MPICXXFLAGS = -g -O2 -w

# Puhti
MPICXXENV = OMPI_CXXFLAGS='' OMPI_CXX='$(HIPCC) --x cu --gpu-architecture=sm_70'
LDFLAGS = -L/appl/spack/v018/install-tree/gcc-11.3.0/openmpi-4.1.4-w2aekq/lib
LIBS = -lmpi

ping-pong: ping-pong.o
	$(HIPCC) $(LDFLAGS) -o $@ $< $(LIBS)

%.o: %.cpp
	$(MPICXXENV) $(MPICXX) $(MPICXXFLAGS) -c -o $@ $<

.PHONY: clean
clean:
	rm -f *.o ping-pong
