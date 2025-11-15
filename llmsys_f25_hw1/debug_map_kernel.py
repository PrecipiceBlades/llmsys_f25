#!/usr/bin/env python3
"""
CUDA MapKernel 调试脚本 - 一元函数测试

mapKernel 只能使用一元函数：
- ID_FUNC (3): 恒等映射
- NEG_FUNC (4): 取负
- SIGMOID_FUNC (7): Sigmoid 激活
- RELU_FUNC (8): ReLU 激活
- LOG_FUNC (10): 对数
- EXP_FUNC (12): 指数
- TANH (18): 双曲正切

使用方法:
    cd /root/llmsys_f25/llmsys_f25_hw1
    uv run python debug_map_kernel.py
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from minitorch import Tensor
from minitorch.cuda_kernel_ops import CudaKernelOps

def debug_map_relu():
    """测试 RELU_FUNC - 最常见的一元激活函数"""
    print("=" * 60)
    print("Test 1: RELU_FUNC - ReLU(x) = max(0, x)")
    print("=" * 60)
    
    # 创建简单的输入 - 包含正负数
    data = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    shape = (2, 3)
    
    t = Tensor.make(data, shape, backend=CudaKernelOps)
    
    print(f"Input shape: {t.shape}")
    print(f"Input strides: {t._tensor._strides}")
    print(f"Input matrix:\n{t.to_numpy()}")
    
    # 执行 ReLU (fn_id = RELU_FUNC = 8)
    result = t.relu()
    
    print(f"\nOutput shape: {result.shape}")
    print(f"Output matrix:\n{result.to_numpy()}")
    print(f"Expected: [[0, 0, 0], [1, 2, 3]]")
    print(f"✓ Match: {np.allclose(result.to_numpy(), [[0, 0, 0], [1, 2, 3]])}")

def debug_map_negate():
    """测试 NEG_FUNC - 取负操作"""
    print("\n" + "=" * 60)
    print("Test 2: NEG_FUNC - Negate(x) = -x")
    print("=" * 60)
    
    # 创建简单矩阵
    data = [1.0, 2.0, 3.0, 4.0]
    shape = (2, 2)
    
    t = Tensor.make(data, shape, backend=CudaKernelOps)
    
    print(f"Input shape: {t.shape}")
    print(f"Input matrix:\n{t.to_numpy()}")
    
    # 执行取负 (fn_id = NEG_FUNC = 4)
    result = -t
    
    print(f"\nOutput shape: {result.shape}")
    print(f"Output matrix:\n{result.to_numpy()}")
    print(f"Expected: [[-1, -2], [-3, -4]]")
    print(f"✓ Match: {np.allclose(result.to_numpy(), [[-1, -2], [-3, -4]])}")

def debug_map_transpose():
    """测试 ID_FUNC + 转置（stride 变化）- mapKernel 的重要用例"""
    print("\n" + "=" * 60)
    print("Test 3: ID_FUNC with Transpose (same data, different stride)")
    print("=" * 60)
    
    # 创建 [2, 3] 矩阵
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    shape = (2, 3)
    
    t = Tensor.make(data, shape, backend=CudaKernelOps)
    
    print(f"Input shape: {t.shape}")
    print(f"Input strides: {t._tensor._strides}")
    print(f"Input as matrix:\n{t.to_numpy()}")
    print(f"  [[1, 2, 3],")
    print(f"   [4, 5, 6]]")
    
    # 转置 - 这会改变 stride，但不改变 storage
    # 内部可能会调用 mapKernel with ID_FUNC 来实现
    t_T = t.permute(1, 0)
    
    print(f"\nTransposed shape: {t_T.shape}")
    print(f"Transposed strides: {t_T._tensor._strides}")
    print(f"Transposed as matrix:\n{t_T.to_numpy()}")
    print(f"  [[1, 4],")
    print(f"   [2, 5],") 
    print(f"   [3, 6]]")
    
    # 关键：storage 是同一块内存！只是访问模式不同
    print(f"\n✓ Same storage: {t._tensor._storage is t_T._tensor._storage}")
    print(f"  这就是 mapKernel + strides 的威力：")
    print(f"  - 输入和输出逻辑形状不同 ([2,3] vs [3,2])")
    print(f"  - 但物理存储相同 [1,2,3,4,5,6]")
    print(f"  - 通过不同的 strides 实现不同的访问模式")

def debug_with_small_data():
    """用非常小的数据测试，便于手动验证"""
    print("\n" + "=" * 60)
    print("Test 4: Tiny data for manual verification")
    print("=" * 60)
    
    # 只有4个元素
    data = [1.0, 2.0, 3.0, 4.0]
    shape = (2, 2)
    
    t = Tensor.make(data, shape, backend=CudaKernelOps)
    
    print(f"Input: {data}")
    print(f"Shape: {shape}")
    print(f"As matrix:\n{t.to_numpy()}")
    
    # 测试 sigmoid
    result = t.sigmoid()
    print(f"\nSigmoid result:")
    print(result.to_numpy())
    
    # 手动计算验证
    import math
    expected = [[1/(1+math.exp(-1)), 1/(1+math.exp(-2))],
                [1/(1+math.exp(-3)), 1/(1+math.exp(-4))]]
    print(f"\nExpected:")
    print(np.array(expected))
    
    print(f"\nMatch: {np.allclose(result.to_numpy(), expected)}")

def debug_step_by_step():
    """逐步追踪 kernel 调用"""
    print("\n" + "=" * 60)
    print("Test 5: Step-by-step tracing")
    print("=" * 60)
    
    # 最简单的情况：1D 数组
    data = [1.0, 2.0, 3.0]
    shape = (3,)
    
    t = Tensor.make(data, shape, backend=CudaKernelOps)
    
    print("Step 1: Create tensor")
    print(f"  data: {data}")
    print(f"  shape: {t.shape}")
    print(f"  strides: {t._tensor._strides}")
    print(f"  size: {t._tensor.size}")
    
    print("\nStep 2: Call negate")
    result = -t
    
    print(f"  result data: {result.to_numpy().tolist()}")
    print(f"  expected: {[-1.0, -2.0, -3.0]}")

if __name__ == "__main__":
    print("CUDA MapKernel Debugging - 一元函数测试\n")
    print("mapKernel 只支持一元函数 (单个输入):")
    print("  - ID_FUNC, NEG_FUNC, RELU_FUNC, SIGMOID_FUNC, etc.")
    print("  - 不支持: ADD, MUL, MAX (那些用 zipKernel)\n")
    
    try:
        debug_map_relu()
        debug_map_negate()
        debug_map_transpose()
        debug_with_small_data()
        debug_step_by_step()
        
        print("\n" + "=" * 60)
        print("✅ All debug tests completed!")
        print("=" * 60)
        print("\n提示：如果测试失败，检查：")
        print("  1. CUDA kernel 是否编译? (bash compile_cuda.sh)")
        print("  2. mapKernel 是否正确实现 ASSIGN2_1 部分?")
        print("  3. 索引计算是否正确? (to_index, index_to_position)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

