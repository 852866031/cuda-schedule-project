# Makefile for the nvidia-smi two-case demo

CXX      := nvcc
TARGET   := two_cases
SRC      := two_cases.cu
CXXFLAGS := -O3 -std=c++17 -Wno-deprecated-gpu-targets

.PHONY: all compile run clean

all: compile

compile:
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

run: compile
	./$(TARGET)

clean:
	rm -f $(TARGET)