#!/bin/bash
# Add CUDA toolkit bin directory to PATH for cicc
export PATH="/usr/lib/nvidia-cuda-toolkit/bin:$PATH"

# Ensure output directory exists
mkdir -p minitorch/cuda_kernels

# Compile CUDA kernel
nvcc -o minitorch/cuda_kernels/combine.so --shared src/combine.cu -Xcompiler -fPIC
