# Makefile for 2D FFT assignment

CXX      = /usr/bin/g++
CXXFLAGS = -Wall -g
.DEFAULT_GOAL := threadDFT2d

threadDFT2d:	threadDFT2d.o Complex.o InputImage.o myBarrier.o
	$(CXX) -g -o threadDFT2d threadDFT2d.o Complex.o InputImage.o myBarrier.o -lpthread

clean:
	@rm *.o threadDFT2d Tower-DFT2D.txt Tower-IDFT2D.txt
