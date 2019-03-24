ifndef LINUX
CC=	riscv64-unknown-elf-gcc
OBJDUMP=riscv64-unknown-elf-objdump
AR=	riscv64-unknown-elf-ar
SUFFIX=
else
CC=	riscv64-unknown-linux-gnu-gcc
OBJDUMP=riscv64-unknown-linux-gnu-objdump
AR=	riscv64-unknown-linux-gnu-ar
SUFFIX= -linux
endif

OBJDIR= ./obj$(SUFFIX)/

VPATH=./src

ISA ?= rv64g

ARFLAGS =  rcs
OPTS =	   -O3
LDFLAGS =  -lm
CFLAGS = -static -g \
	-march=$(ISA) \
	-Wa,-march=$(ISA)xhwacha \
	-Wl,--gc-sections \
	-Wall \
	-Wno-comment \
	-Wno-unknown-pragmas \
	-Wno-misleading-indentation \
	-Wfatal-errors \
	-fPIC \
	-ffast-math \
	-fno-common \
	-ffunction-sections \
	-fdata-sections \

CFLAGS += -Iinclude
CFLAGS += $(OPTS)

ifdef SCALAR
CFLAGS += -DUSE_SCALAR
SUFFIX := -scalar$(SUFFIX)
endif

COMMON_FILES = \
	util \
	util_asm \
	layer \
	convolutional_layer \
	maxpool_layer \
	gemm \
	gemm_asm \
	fc_layer \
	fc_layer_asm \
	parse_args
OBJS = $(addprefix $(OBJDIR), $(addsuffix .o, $(COMMON_FILES)))
DEPS = $(wildcard include/*.h) Makefile $(OBJDIR)
EXECS = $(addsuffix $(SUFFIX), \
	tiny_yolo_16 \
	tiny_yolo_32 \
	test \
	squeezenet_32 \
	squeezenet_encoded_32 \
	squeezenet_encoded_compressed_32 \
	alexnet_32 \
	alexnet_encoded_32)
EXECOBJS = $(addsuffix .o, $(addprefix $(OBJDIR), $(EXECS)))
DUMPS = $(addsuffix .dump, $(EXECS))

default : all
all : $(EXECS) $(DUMPS)

%.dump: %
	$(OBJDUMP) -d $^ > $@

$(EXECS): %$(SUFFIX): $(OBJDIR)%.o $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.S $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $@

.PHONY: clean

clean:
	rm -rf $(EXECS) $(DUMPS) $(OBJDIR)

