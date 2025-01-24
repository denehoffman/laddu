SHELL := /bin/bash
INC_DIR := -I$(WORKDIR)/libraries
LIB_DIR := -L$(WORKDIR)/lib

include $(AMPTOOLS_HOME)/Makefile.settings

INC_DIR += -I$(AMPTOOLS) -I$(shell root-config --incdir)
CXX_FLAGS += $(shell root-config --cflags)


# need some linking information available
LIB_DIR += -L$(shell root-config --libdir) -L$(AMPTOOLS)/lib

LIBS :=  -lBenchmark$(SUFFIX) -lAmpTools	$(EXTRA_LIBS) $(shell root-config --glibs) -lstdc++

NOMPI_SRC := $(shell ls *.cc | grep -vi mpi)

TARGET_EXE := $(NOMPI_SRC:%.cc=%$(SUFFIX))

.PHONY: default bin
.PRECIOUS: .%$(SUFFIX).o

default: bin $(TARGET_EXE)

bin:
	$(Q)-mkdir $(INSTALL_BIN) >& /dev/null ||:

# for the non-MPI executables we want to strip out the -DUSEMPI flag
# from the compiler arguments and also be sure we link against the
# non-MPI version of AmpTools

$(NOMPI_SRC:%.cc=%$(SUFFIX)): %$(SUFFIX) : .%$(SUFFIX).o
	$(vecho) "--> Linking $*$(SUFFIX)"
	$(Q)$(CXX) $(subst -DUSEMPI,,$(CXX_FLAGS)) -o $(INSTALL_BIN)/$*$(SUFFIX) $< 	$(INC_DIR) $(LIB_DIR) $(subst _MPI,,$(LIBS))

.%$(SUFFIX).o: %.cc .%$(SUFFIX).d
	$(vecho) "-> Compiling $<"
	$(Q)$(CXX) $(subst -DUSEMPI,,$(CXX_FLAGS)) -M -MP -MT $@ -o .$*$(SUFFIX).d $< $(INC_DIR)
	$(Q)$(CXX) $(subst -DUSEMPI,,$(CXX_FLAGS)) -c -o $@ $< $(INC_DIR)

DEPFILES := $(NOMPI_SRC:%.cc=.%$(SUFFIX).d)
$(DEPFILES):

clean:
	$(Q)rm -f .*.o .*.d $(TARGET_EXE)

-include $(DEPFILES)
