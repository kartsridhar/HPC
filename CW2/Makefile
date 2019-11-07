#
# Makefile to build stencil MPI program
#

CC=mpiicc

CFLAGS=-Wall -Ofast

EXES=stencil.exe


stencil.exe: stencil.c
	$(CC) -std=c99 $(CFLAGS) -o $@ $^

all: $(EXES)

.PHONY: clean all

clean:
	\rm -f $(EXES)
	\rm -f *.o
