SHELL=/bin/bash

CC=gcc
CCFLAGS = -O0
MPICC=mpicc

CMD = jacobi jacobi1 jacobi2
SRC = $(CMD:=.c)
OBJ = $(SRC:.c=.o) 

all: $(CMD)

jacobi: jacobi.o
	$(CC) $(CCFLAGS) -o $@ $< -lm

jacobi1: jacobi1.o
	$(MPICC) $(CCFLAGS) -o $@ $<

jacobi2: jacobi2.o
	$(MPICC) $(CCFLAGS) -o $@ $<

jacobi.o: jacobi.c
	$(CC) $(CCFLAGS) -c $<

jacobi1.o: jacobi1.c
	$(MPICC) $(CCFLAGS) -c $<

jacobi2.o: jacobi2.c
	$(MPICC) $(CCFLAGS) -c $<

clean:
	rm $(OBJ) $(CMD)
