OBJDIR= ./obj/

CC=	/scratch/jerryz/hwacha-build/build-tools/bin/riscv64-unknown-elf-gcc
OBJDUMP=/scratch/jerryz/hwacha-build/build-tools/bin/riscv64-unknown-elf-objdump
AR=	/scratch/jerryz/hwacha-build/build-tools/bin/riscv64-unknown-elf-ar

VPATH=./src


ARFLAGS =  rcs
OPTS =	   -O3
LDFLAGS =  -lm
COMMON =   -Iinclude/
CFLAGS =   -Wall -Wno-comment -Wno-unknown-pragmas -Wno-misleading-indentation -Wfatal-errors -fPIC -march=RV64IMAFDXhwacha -ffast-math -static -fno-common -ffunction-sections -fdata-sections -Wl,--gc-sections -s
CFLAGS +=  $(OPTS)

OBJ = util.o layer.o util_asm.o convolutional_layer.o maxpool_layer.o gemm.o gemm_asm.o fc_layer.o fc_layer_asm.o
OBJS = $(addprefix $(OBJDIR), $(OBJ))

DEPS = $(wildcard include/*.h) Makefile obj

EXECS = tiny_yolo_16 tiny_yolo_32 test squeezenet_32 squeezenet_encoded_32 squeezenet_encoded_compressed_32 alexnet_32 alexnet_encoded_32
EXECOBJS = $(addsuffix .o, $(addprefix $(OBJDIR), $(EXECS)))
DUMPS = $(addsuffix .dump, $(EXECS))

default : all
all : $(EXECS) $(DUMPS)

%.dump : %
	$(OBJDUMP) -d $^ > $@


tiny_yolo_16: ./obj/tiny_yolo_16.o $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

tiny_yolo_32: ./obj/tiny_yolo_32.o $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

test: ./obj/test.o $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

squeezenet_32: ./obj/squeezenet_32.o $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

squeezenet_encoded_32: ./obj/squeezenet_encoded_32.o $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

squeezenet_encoded_compressed_32: ./obj/squeezenet_encoded_compressed_32.o $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

alexnet_32: ./obj/alexnet_32.o $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

alexnet_encoded_32: ./obj/alexnet_encoded_32.o $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.S $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

obj:
	mkdir -p obj

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXECS) $(DUMPS) $(EXECOBJS)

