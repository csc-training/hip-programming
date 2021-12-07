HIPCC = hipcc
HIPFLAGS=--offload-arch=gfx908
SRC1 = copy.cpp
EXE1 = copy
SRC2 = async_case2.cpp
EXE2 = async_case2

.SUFFIXES:


all: $(EXE1) $(EXE2)


copy: $(EXE1)

$(EXE1): $(SRC1)
	$(HIPCC) $(HIPFLAGS) $(INCLUDES)  -o $@ $<


async_case2: $(EXE2)

$(EXE2): $(SRC2)
	$(HIPCC) $(HIPFLAGS) $(INCLUDES)  -o $@ $<


.PHONY: clean

clean:
	rm -f  $(EXE1) $(EXE2) 
