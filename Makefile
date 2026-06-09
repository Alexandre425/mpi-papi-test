CXX = mpicxx
CXXFLAGS = -O3 -fopenmp $(shell pkg-config --cflags papi)
LDFLAGS = $(shell pkg-config --libs papi)

all: daxpy-scalar daxpy-avx2

daxpy-scalar: daxpy.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

daxpy-avx2: daxpy.cpp
	$(CXX) $(CXXFLAGS) -mavx2 -o $@ $^ $(LDFLAGS)