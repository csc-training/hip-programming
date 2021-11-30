HIPCC = hipcc
HIPFLAGS=--offload-arch=gfx908
SRC = async_serial.cpp
EXE = async_serial


.SUFFIXES:

all: $(EXE)

$(EXE): $(SRC)
	$(HIPCC) $(HIPFLAGS) $(INCLUDES)  -o $@ $<

.PHONY: clean

clean:
	rm -f  $(EXE) 
