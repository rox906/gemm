OBJ = gemm
OPT = -std=c++11 -lcublas -arch=sm_70

ifdef DEBUG
OPT += -g -G
endif

all : $(OBJ)

gemm : gemm.cu
	nvcc $^ -o $@ $(OPT)

.PHONY : clean
clean :
	rm -rf $(OBJ)