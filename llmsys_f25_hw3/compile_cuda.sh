#!/bin/bash
mkdir -p minitorch/cuda_kernels
# Add CUDA toolkit bin to PATH for cicc
export PATH="/usr/lib/nvidia-cuda-toolkit/bin:$PATH"
nvcc -ccbin gcc-12 -o minitorch/cuda_kernels/combine.so --shared src/combine.cu -Xcompiler -fPIC
