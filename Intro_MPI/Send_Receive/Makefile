#
# Makefile to build example MPI programs
#

CC=mpiicc

CFLAGS=-Wall

EXES=send_recv.exe

all: $(EXES)

$(EXES): %.exe : %.c
	$(CC) $(CFLAGS) -o $@ $^

.PHONY: clean all

clean:
	\rm -f $(EXES)
	\rm -f *.o