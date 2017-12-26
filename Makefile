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

OBJ = util.o layer.o util_asm.o convolutional_layer.o maxpool_layer.o gemm.o gemm_asm.o
OBJS = $(addprefix $(OBJDIR), $(OBJ))

DEPS = $(wildcard include/*.h) Makefile obj

TINYYOLO_16_OBJ = $(addprefix $(OBJDIR), tiny_yolo_16.o)
TINYYOLO_16 = tiny_yolo_16

TINYYOLO_32_OBJ = $(addprefix $(OBJDIR), tiny_yolo_32.o)
TINYYOLO_32 = tiny_yolo_32

TEST_OBJ = $(addprefix $(OBJDIR), test.o)
TEST = test

EXECS = tiny_yolo_16 tiny_yolo_32 test
DUMPS = $(addpostfix .dump, $(EXECS))

all : $(EXECS) $(DUMPS)

%.dump : %
	$(OBJDUMP) -d $^ > $@


$(TINYYOLO_16): $(TINYYOLO_16_OBJ) $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(TEST): $(TEST_OBJ) $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(TINYYOLO_32): $(TINYYOLO_32_OBJ) $(OBJS)
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

