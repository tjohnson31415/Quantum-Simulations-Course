EXECUTABLE:= pathIntegrator

CUDA_PATH ?= /usr/local/cuda

NVCC ?= nvcc
NVCCFLAGS = -arch=compute_30 -code=sm_30,compute_30
CXX = g++
CXXFLAGS = -O3 -g

INCPATH = -I$(CUDA_INC_PATH) -I$(CULA_INC_PATH)
LIBPATH = -L$(CUDA_LIB_PATH_64) -L$(CULA_LIB_PATH_64)
LIBS = -lcudart -lcublas -lcufft -lcula_lapack -lglut -lGLEW -lGL -lGLU 

# files
CPP_SOURCES       := 
CU_SOURCES        := pathIntegrator.cu
HEADERS           := $(wildcard *.h)
CPP_OBJS          := $(patsubst %.cpp, %.o, $(CPP_SOURCES))
CU_OBJS           := $(patsubst %.cu, %.cu_o, $(CU_SOURCES))

all: $(EXECUTABLE)

%.cu_o : %.cu
	$(NVCC) -c $(NVCCFLAGS) $(INCPATH) -o $@ $<

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o $@ $<

$(EXECUTABLE): $(CPP_OBJS) $(CU_OBJS)
	$(CXX) -o $(EXECUTABLE) $(CU_OBJS) $(CPP_OBJS) $(LDFLAGS) $(INCPATH) $(LIBPATH) $(LIBS)

clean:
	rm -f *.o *.cu_o
