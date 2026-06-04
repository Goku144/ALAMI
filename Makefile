##############
# PUBLIC VARS
##############

BASEPATH :=
PROG := dl
INC := -Ipublic/inc -Ilib/src -I/usr/local/cuda-13.2/include -I/usr/local/cuda-13.2/include/cccl -I/usr/local/cuda-13.2/targets/x86_64-linux/include -I/usr/include

###############
# NVCC OPTIONS 
###############

NVCC ?= nvcc
CUDA_AVAILABLE := $(if $(NVCC),1,0)
FLAG_CUDA := -O3 -Wno-deprecated-gpu-targets -arch=sm_75 -Xcompiler -fno-exceptions -diag-suppress 550
LIBS_CUDA := -lcudnn -lcusparse -lcusolver -lcurand -lcublasLt -lcublas -lcudart
SRCS_CUDA := $(shell find lib/src -name '*.cu')
OBJS_CUDA := $(patsubst lib/src/%.cu, lib/bin/%.o, $(SRCS_CUDA))

#############
# CONDITIONS
#############

ifeq ($(CUDA_AVAILABLE),0)
$(info visit the website to install: https://developer.nvidia.com/cuda/toolkit)
$(error because CUDA is not available compiling .cu sources)
endif

############
# Build App
############

ml:
	@python3 app/src/ml.py

$(PROG): app/build/$(PROG)
	@mkdir -p public/checkpoints
	@./$<

app/build/$(PROG): app/bin/$(PROG).o $(OBJS_CUDA)
	@mkdir -p $(dir $@)
	@$(NVCC) $^ $(LIBS_CUDA) -o $@

app/bin/$(PROG).o: app/src/$(PROG).cu
	@mkdir -p $(dir $@)
	@$(NVCC) $(FLAG_CUDA) $(INC) -c $< -o $@

############
# Build Lib
############

lib: $(OBJS_CUDA)

lib/bin/%.o: lib/src/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(FLAG_CUDA) $(INC) -c $< -o $@

############
# Utility
############

dataset:
	@python3 public/target/src/DataSet.py

clean:
	rm -rf lib/bin $(PROG)/bin $(PROG)/build

.PHONY: $(PROG) run lib clean
