#
# makefile for my CUDA hotplate
#

SRC = src/hotplate.cu
HEADER = include/hotplate.h
OBJ = bin/hotplate.o
EXE = bin/Hotplate
CUDA_PATH = /usr/local/cuda
LIBDIR = $(CUDA_PATH)/lib

CC = /usr/local/cuda/bin/nvcc
FLAGS = -Iinclude -I$(CUDA_PATH)/include -L$(LIBDIR) -O3 -g

bin: $(OBJ)
	$(CC) $(FLAGS) $(OBJ) -o $(EXE)

$(OBJ): $(SRC) $(HEADER) makefile
	$(CC) $(FLAGS) -c $(SRC) -o $(OBJ)

runtest: bin
	-@echo "CUDA path: $(CUDA_PATH)"
	-@echo "CUDA libdir: $(LIBDIR)"
	$(EXE)

tags: $(SRC) $(HEADER)
	ctags src/* include/*

prep:
	@if [ ! -d bin ]; then echo "creating bin dir..."; mkdir bin; else echo "bin ok..."; fi

package:
	tar -cvzf cuda_hotplate.tgz src include makefile

clean:
	@-rm -rfv bin/*

