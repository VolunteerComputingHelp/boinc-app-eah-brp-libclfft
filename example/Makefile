PWD = $(shell pwd)
NVIDIA_SDK_INSTALL_PATH ?= /usr/local/cuda

TARGET = clfft_example

CXXFLAGS += -O3 -Wall -g -I$(NVIDIA_SDK_INSTALL_PATH)/OpenCL/common/inc -I../include
LDFLAGS += $(PWD)/../lib/libclfft.a

OS = $(shell uname -s)
ifeq ($(OS), Darwin)
	LDFLAGS += -framework OpenCL
else
	LDFLAGS += -lOpenCL
endif


$(TARGET): main.cpp
	g++ $(CXXFLAGS) $(LDFLAGS) -o $(TARGET) main.cpp

clean:
	rm -f $(TARGET)
