#
# Makefile to build stencil MPI program
#

CC=mpiicc

CFLAGS=-Wall

OFLAGS=-O3 -xHOST

OMP=-fopenmp

EXES=stencil.exe


stencil.exe: stencil.c
	$(CC) -std=c99 $(OMP) $(CFLAGS) $(OFLAGS) -o $@ $^

all: $(EXES)

.PHONY: clean all

clean:
	\rm -f $(EXES)
	\rm -f *.o
