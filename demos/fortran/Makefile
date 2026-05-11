FC      = ftn
HIPCC   = hipcc

FFLAGS  = -O3
HIPFLAGS = -O3
HIPFLAGS += --offload-arch=gfx90a

TARGET  = saxpy

FOBJS   = main.o
COBJS   = saxpy.o

all: $(TARGET)

$(TARGET): $(FOBJS) $(COBJS)
	$(FC) -o $@ $(FOBJS) $(COBJS) -lamdhip64

main.o: main.f90
	$(FC) $(FFLAGS) -c $<

saxpy.o: saxpy.cpp
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

clean:
	rm -f *.o *.mod $(TARGET)
