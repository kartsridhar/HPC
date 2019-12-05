#
# Makefile to build stencil MPI program
#

CC=mpiicc

CFLAGS=-Wall

OFLAGS=-O3 -xHost

OMP=-fopenmp

EXES=stencil.exe

REPORT=-qopt-report=5

stencil.exe: stencil.c
	$(CC) -std=c99 $(OMP) $(CFLAGS) $(OFLAGS) -o $@ $^

all: $(EXES)

.PHONY: clean all

clean:
	\rm -f $(EXES)
	\rm -f *.o
