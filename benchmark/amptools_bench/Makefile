SHELL := /bin/bash
WORKDIR := $(shell pwd)
INSTALL_BIN := $(WORKDIR)/bin
INSTALL_LIB := $(WORKDIR)/lib

export

.PHONY: default clean force

default: compiler_flags
	@echo "=== BUILDING AMPLITUDES ==="
	@$(MAKE) -C $(WORKDIR)/libraries
	@$(MAKE) -C $(WORKDIR)/executables

clean:
	@$(MAKE) -C $(WORKDIR)/libraries clean
	@$(MAKE) -C $(WORKDIR)/executables clean

compiler_flags: force
	@echo '$(CXX_FLAGS)' | cmp -s - $@ || echo '$(CXX_FLAGS)' > $@
