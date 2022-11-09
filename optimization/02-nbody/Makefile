HIPCC = hipcc
HIPFLAGS=--gpu-architecture=sm_70
SRC1 = BarnzNhutt_optimal.cpp
EXE1 = nbody

.SUFFIXES:


all: $(EXE1)


nbody: $(EXE1)

$(EXE1): $(SRC1)
	$(HIPCC) $(HIPFLAGS) $(INCLUDES)  -o $@ $<



.PHONY: clean

clean:
	rm -f  $(EXE1) 
