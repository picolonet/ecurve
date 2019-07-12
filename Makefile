CXX ?= g++
GENCODES ?= 60
GENCODES = 61

ifdef GMP_HOME
  GMP_INC := -I$(GMP_HOME)/include
  GMP_LIB := -L$(GMP_HOME)/lib
endif
ifndef GMP_HOME
  GMP_INC :=
  GMP_LIB :=
endif

INCLUDE_DIRS = -I./src -I./ff -L./ff/build/libff/ $(GMP_INC) $(GMP_LIB)
# NVCC_FLAGS = -src-in-ptx -keep -ccbin $(CXX) -std=c++11 -Xcompiler -Wall,-Wextra -g -G -DUSE_GPU=1
NVCC_FLAGS = -ccbin $(CXX) -std=c++11 -Xcompiler -Wall,-Wextra -g -G -DUSE_GPU=1
NVCC_OPT_FLAGS = -DNDEBUG  
NVCC_TEST_FLAGS = -lineinfo
NVCC_DBG_FLAGS = -g -G 
NVCC_LIBS = -lstdc++ -lgmp -lff -lgomp -lprocps
NVCC_TEST_LIBS = -lgtest

all:
	@echo "Please run 'make check' or 'make bench'."

tests/test-suite: tests/test-suite.cu
	nvcc $(NVCC_TEST_FLAGS) $(NVCC_FLAGS) $(GENCODES:%=--gpu-architecture=compute_%) $(GENCODES:%=--gpu-code=sm_%) $(INCLUDE_DIRS) $(NVCC_LIBS) $(NVCC_TEST_LIBS) -o $@ $<

check: tests/test-suite
	@./tests/test-suite

bench/bench: bench/bench.cu
	nvcc $(NVCC_OPT_FLAGS) $(NVCC_FLAGS) $(GENCODES:%=--gpu-architecture=compute_%) $(GENCODES:%=--gpu-code=sm_%) $(INCLUDE_DIRS) $(NVCC_LIBS) -o $@ $<

bench: bench/bench

main: main.cu quad_mul.cu
	nvcc $(NVCC_OPT_FLAGS) $(NVCC_FLAGS) $(GENCODES:%=--gpu-architecture=compute_%) $(GENCODES:%=--gpu-code=sm_%) $(INCLUDE_DIRS) $(NVCC_LIBS) -o $@ $<

play: play.cu  play_mul.cu
	nvcc $(NVCC_OPT_FLAGS) $(NVCC_FLAGS) $(GENCODES:%=--gpu-architecture=compute_%) $(GENCODES:%=--gpu-code=sm_%) $(INCLUDE_DIRS) $(NVCC_LIBS) -o $@ $<

p2: p2.cu p2_mul.cu
	nvcc $(NVCC_OPT_FLAGS) $(NVCC_FLAGS) $(GENCODES:%=--gpu-architecture=compute_%) $(GENCODES:%=--gpu-code=sm_%) $(INCLUDE_DIRS) $(NVCC_LIBS) -o $@ $<

myfq_test: myfq_test.cu utils.cu myfq.cu
	nvcc $(NVCC_OPT_FLAGS) $(NVCC_FLAGS) $(GENCODES:%=--gpu-architecture=compute_%) $(GENCODES:%=--gpu-code=sm_%) $(INCLUDE_DIRS) $(NVCC_LIBS) -o $@ $<

ec: ec.cu utils.cu myfq.cu myg1.cu
	nvcc $(NVCC_OPT_FLAGS) $(NVCC_FLAGS) $(GENCODES:%=--gpu-architecture=compute_%) $(GENCODES:%=--gpu-code=sm_%) $(INCLUDE_DIRS) $(NVCC_LIBS) -o $@ $<

add_exp: add_exp.cu 
	nvcc $(NVCC_OPT_FLAGS) $(NVCC_FLAGS) $(GENCODES:%=--gpu-architecture=compute_%) $(GENCODES:%=--gpu-code=sm_%) $(INCLUDE_DIRS) $(NVCC_LIBS) -o $@ $<

main2: main.cu new_mul.cu
	nvcc $(NVCC_OPT_FLAGS) $(NVCC_FLAGS) $(INCLUDE_DIRS) $(NVCC_LIBS) -o $@ $<

.PHONY: clean
clean:
	$(RM) tests/test-suite bench/bench
