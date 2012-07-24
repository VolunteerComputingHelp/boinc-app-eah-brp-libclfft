PWD = $(shell pwd)
NVIDIA_SDK_INSTALL_PATH ?= /usr/local/cuda
AMDAPPSDKROOT ?= /opt/AMDAPP
CXX ?= g++

TARGET = clfft_example

CXXFLAGS += -O3 -Wall -g -I$(NVIDIA_SDK_INSTALL_PATH)/OpenCL/common/inc -I$(AMDAPPSDKROOT)/include -I../include
LDFLAGS += $(PWD)/../lib/libclfft.a

OS = $(shell uname -s)
ARCH = $(shell uname -m)
ifneq ($(ARCH), x86_64)
        ARCH = x86
endif
ifeq ($(OS), Darwin)
	LDFLAGS += -framework OpenCL
	LDFLAGS += -framework Accelerate
else
	LDFLAGS += -L$(AMDAPPSDKROOT)/lib/$(ARCH)
	LDFLAGS += -lOpenCL
endif


$(TARGET): main.cpp Makefile $(PWD)/../lib/libclfft.a
	$(CXX) $(CXXFLAGS) main.cpp $(LDFLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET)
