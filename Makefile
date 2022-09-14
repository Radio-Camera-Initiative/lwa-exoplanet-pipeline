CC = /usr/bin/g++
FLAGS = -std=c++2a -g

LDFLAGS = -lrt -Wall -lpthread -lidg-cuda -L/opt/lib -lidg-util -lidg-common -lidg -lcasa_ms -lcasa_tables -lcasa_casa

CUDA_PATH       ?= /usr/local/cuda-11.2
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib
LENDER_INC_PATH     ?= ../rci-memory-lender
CALIBRATION_INC_PATH    ?= ../calibration-application

# CUDA code generation flags
GENCODE_FLAGS   := -gencode arch=compute_35,code=sm_35 \
        -gencode arch=compute_50,code=sm_50 \
        -gencode arch=compute_52,code=sm_52 \
        -gencode arch=compute_60,code=sm_60 \
        -gencode arch=compute_61,code=sm_61 \
        -gencode arch=compute_61,code=compute_61 \
	-gencode arch=compute_75,code=sm_75
        
# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc

# OS-specific build flags
ifeq ($(shell uname),Darwin)
	LDFLAGS       := $(LDFLAGS) -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart -lcufft
	CCFLAGS   	  := -arch $(OS_ARCH)
else
	ifeq ($(OS_SIZE),32)
		LDFLAGS   := $(LDFLAGS) -L$(CUDA_LIB_PATH) -lcudart
		CCFLAGS   := -m32
	else
		CUDA_LIB_PATH := $(CUDA_LIB_PATH)64
		LDFLAGS       := $(LDFLAGS) -L$(CUDA_LIB_PATH) -lcudart
		CCFLAGS       := -m64
	endif
endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
	NVCCFLAGS := -m32
else
	NVCCFLAGS := -m64
endif

NVCCFLAGS += --compiler-bindir $(CC)


INCLUDE = -I. -I/opt/include -I$(CUDA_INC_PATH) -I$(LENDER_INC_PATH) -I$(CALIBRATION_INC_PATH)

SOURCES = gridding.cpp apply.o

TARGETS = gridding

all: gridding apply.o

gridding: $(SOURCES)
	$(CC) $(FLAGS) $(SOURCES) -o $(TARGETS) -O3 $(LDFLAGS) $(INCLUDE)

apply.o: /fastpool/mlaures/calibration-application/calibration.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $<

# $(TARGETS): r3.o main.o
# 	$(CC) $(FLAGS) $^ -o $@ -O3 $(LDFLAGS)

clean:
	rm -f *.o $(TARGETS)

again: clean $(TARGETS)
