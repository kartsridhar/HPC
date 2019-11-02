#
# Makefile to build stencil MPI program
#

CC=mpiicc

CFLAGS=-Wall

EXES=stencil.exe


stencil.exe: stencil.c
	$(CC) $(CFLAGS) -o $@ $^

all: $(EXES)

.PHONY: clean all

clean:
	\rm -f $(EXES)
	\rm -f *.o