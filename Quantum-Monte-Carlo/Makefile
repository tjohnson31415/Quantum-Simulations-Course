EXECUTABLE:= finalproject

BUILDDIR:= build

CUDA_ROOT = /usr/cuda-5.5

INCPATH = -I$(CUDA_ROOT)/include
LIBPATH = -L$(CUDA_ROOT)/lib64

LIBS = -lcudart -lgomp

NVCC = $(CUDA_ROOT)/bin/nvcc
NVCCFLAGS = -arch=compute_35 -code=sm_35,compute_35 -Xcompiler -Wall

CXX = g++
CXXFLAGS = -O3 -Wall -fopenmp

# List of files contained in the project
CPP_SOURCES       := main.cpp

CU_SOURCES        := MetropolisSampler.cu \
                     PathAnalyzer.cu \
                     ThrustHistogram.cu

# Generate names for object files and such
HEADERS           := $(wildcard *.h)
CPP_OBJS          := $(patsubst %.cpp, $(BUILDDIR)/%.o,    $(CPP_SOURCES))
CU_OBJS           := $(patsubst %.cu,  $(BUILDDIR)/%.cu_o, $(CU_SOURCES))
TEST_OBJS         := $(patsubst %.cu,  $(BUILDDIR)/%.cu_o, $(TEST_SOURCES))

# Creates some directories for organization
$(shell mkdir -p $(BUILDDIR))

# Make targets list

.PHONY: all
all: $(EXECUTABLE)

$(BUILDDIR)/%.cu_o : %.cu %.h
	$(NVCC) -c $(NVCCFLAGS) $(INCPATH) -o $@ $<

$(BUILDDIR)/%.cu_o : %.cu
	$(NVCC) -c $(NVCCFLAGS) $(INCPATH) -o $@ $<

$(BUILDDIR)/%.o: %.cpp %.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o $@ $<

$(BUILDDIR)/%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o $@ $<

$(EXECUTABLE): $(CPP_OBJS) $(CU_OBJS)
	$(CXX) -o $(EXECUTABLE) $(CU_OBJS) $(CPP_OBJS) $(LDFLAGS) $(INCPATH) $(LIBPATH) $(LIBS)

.PHONY: clean
clean:
	rm -f $(BUILDDIR)/*.o $(BUILDDIR)/*.cu_o
