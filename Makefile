OBJDIR= ./obj/

CC=	/scratch/jerryz/hwacha-build/build-tools/bin/riscv64-unknown-elf-gcc
OBJDUMP=/scratch/jerryz/hwacha-build/build-tools/bin/riscv64-unknown-elf-objdump
AR=	/scratch/jerryz/hwacha-build/build-tools/bin/riscv64-unknown-elf-ar

VPATH=./src


ARFLAGS =  rcs
OPTS =	   -O3
LDFLAGS =  -lm
COMMON =   -Iinclude/
CFLAGS =   -Wall -Wno-comment -Wno-unknown-pragmas -Wno-misleading-indentation -Wfatal-errors -fPIC -march=RV64IMAFDXhwacha -ffast-math -static -fno-common -g
CFLAGS +=  $(OPTS)

OBJ = util.o layer.o util_asm.o
OBJS = $(addprefix $(OBJDIR), $(OBJ))

DEPS = $(wildcard include/*.h) Makefile obj

TINYYOLO_OBJ = $(addprefix $(OBJDIR), tiny_yolo.o)
TINYYOLO = tiny_yolo

all : $(TINYYOLO) $(TINYYOLO).dump

%.dump : %
	$(OBJDUMP) -d $^ > $@

$(TINYYOLO): $(TINYYOLO_OBJ) $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.S $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

obj:
	mkdir -p obj

.PHONY: clean

clean:
	rm -rf $(OBJS) $(TINYYOLO) $(TINYYOLO_OBJ)

