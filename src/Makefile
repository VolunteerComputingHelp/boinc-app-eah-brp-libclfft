NVIDIA_SDK_INSTALL_PATH ?= /usr/local/cuda

TARGET = libclfft.a

CXXFLAGS += -O3 -Wall -g -I$(NVIDIA_SDK_INSTALL_PATH)/OpenCL/common/inc -I../include
OS = $(shell uname -s)
ifeq ($(OS), Darwin)
	CXXFLAGS += -I/System/Library/Frameworks
endif

OBJECTS = fft_setup.o fft_execute.o fft_kernelstring.o


$(TARGET): $(OBJECTS)
	ar rcs $(TARGET) $(OBJECTS)
	mkdir -p ../lib
	cp $(TARGET) ../lib/

fft_setup.o: fft_setup.cpp fft_internal.h fft_base_kernels.h
	g++ $(CXXFLAGS) -c fft_setup.cpp
	
fft_execute.o: fft_execute.cpp ../include/clFFT.h fft_internal.h
	g++ $(CXXFLAGS) -c fft_execute.cpp

fft_kernelstring.o: fft_kernelstring.cpp ../include/clFFT.h fft_internal.h
	g++ $(CXXFLAGS) -c fft_kernelstring.cpp

clean:
	rm -f *.o
	rm -f *.a
