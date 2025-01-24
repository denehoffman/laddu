SHELL := /bin/bash
SRCFILES := $(wildcard *.cc)

.PHONY: default clean
.PRECIOUS: .%$(SUFFIX).o

default: $(LIB)

%.a:  $(SRCFILES:%.cc=.%$(SUFFIX).o)
	$(vecho) "--> Archiving $@"
	$(Q)ar -rsc $@ $^

.%$(SUFFIX).o: %.cc .%$(SUFFIX).d
	$(vecho) "-> Compiling $<"
	$(Q)$(CXX) $(CXX_FLAGS) -M -MP -MT $@ -o .$*$(SUFFIX).d $< $(INC_DIR)
	$(Q)$(CXX) $(CXX_FLAGS) -c -o $@ $< $(INC_DIR)

.%$(SUFFIX).o: %.cu .%$(SUFFIX).d
	$(vecho) "-> Compiling $<"
	$(Q)$(NVCC) $(NVCC_FLAGS) -M -MT $@ -o .$*$(SUFFIX).d $< $(INC_DIR)
	$(Q)$(NVCC) $(NVCC_FLAGS) -c -o $@ $< $(INC_DIR)

DEPFILES := $(SRCFILES:%.cc=.%$(SUFFIX).d)
$(DEPFILES):

clean:
	$(Q)-rm -f .*.o .*.d *.a

-include $(DEPFILES)
