TARGET = clfft_example

CXXFLAGS += -O3 -Wall -g -I../include
LDFLAGS += ../lib/libclfft.a

OS = $(shell uname -s)
ifeq ($(OS), Darwin)
	LDFLAGS += -framework OpenCL
endif


$(TARGET): main.cpp
	g++ $(CXXFLAGS) $(LDFLAGS) -o $(TARGET) main.cpp

clean:
	rm -f $(TARGET)
