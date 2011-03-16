PWD = $(shell pwd)
NVIDIA_SDK_INSTALL_PATH ?= /usr/local/cuda
ATISTREAMSDKROOT ?= /usr/local/ati

TARGET = clfft_example

CXXFLAGS += -O3 -Wall -g -I$(NVIDIA_SDK_INSTALL_PATH)/OpenCL/common/inc -I$(ATISTREAMSDKROOT)/include -I../include
LDFLAGS += $(PWD)/../lib/libclfft.a

OS = $(shell uname -s)
ARCH = $(shell uname -m)
ifeq ($(OS), Darwin)
	LDFLAGS += -framework OpenCL
	LDFLAGS += -framework Accelerate
else
	LDFLAGS += -L$(ATISTREAMSDKROOT)/lib/$(ARCH)
	LDFLAGS += -lOpenCL
endif


$(TARGET): main.cpp Makefile $(PWD)/../lib/libclfft.a
	g++ $(CXXFLAGS) main.cpp $(LDFLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET)
