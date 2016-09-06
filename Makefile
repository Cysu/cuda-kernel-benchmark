CPP_HEADERS = $(wildcard *.hpp)
CPP_SOURCES = $(wildcard *.cpp)
CU_SOURCES = $(wildcard *.cu)
OBJECTS = $(CPP_SOURCES:.cpp=.o)
BINS = $(CU_SOURCES:.cu=)

CC_FLAGS = -O3 -Wall -Wextra -m64
CUDA_ARCH := -gencode arch=compute_20,code=sm_20 \
		-gencode arch=compute_20,code=sm_21 \
		-gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_50,code=compute_50
NVCC_FLAGS = -O3 $(CUDA_ARCH) -Xcompiler -Wall -m64

all: $(BINS) Makefile

%: %.cu $(OBJECTS) $(CPP_HEADERS)
	nvcc $(NVCC_FLAGS) -o $@ $<

%.o: $.cpp $(CPP_HEADERS)
	g++ -c $(CC_FLAGS) -I $(CUDA_HOME)/include $<

clean:
	rm -f $(OBJECTS) $(BINS)