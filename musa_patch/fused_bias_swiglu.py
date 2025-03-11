import torch
import torch.nn.functional as F

def bias_swiglu_impl(input, bias=None, fp8_input_store=False):
    return F.SwishGLU(input)

import sys
for k in sys.modules:
    if k.startswith('megatron.core.fusions.fused_bias_swiglu'):
        print(f'k is {k}')
        for target in ['bias_swiglu_impl']:
            if getattr(sys.modules[k], target, None):
                print(f'target is {target}')
                setattr(sys.modules[k], target, bias_swiglu_impl)
