from .autodiff import * 
from .datasets import datasets
from .module import *  
from .nn import *  
from .optim import *  

from .tensor import *  
from .tensor_data import *  
from .tensor_functions import *  
from .tensor_ops import *  
from .testing import MathTest, MathTestVariable

# Try to import CUDA ops if available
try:
    from .cuda_kernel_ops import CudaKernelOps
except ImportError:
    pass

version = "0.4"
