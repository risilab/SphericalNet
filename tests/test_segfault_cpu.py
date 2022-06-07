import torch

import sys
sys.path.insert(1, '../models/')
import gelib_torchC as gelib

torch.manual_seed(123)

# ---- CG-product ---------------------------------------------------------------------------------------------
# In a full CG-product each fragment in each part of x is multiplied with each fragment of each part in y 

# Define the type 
batch = 1
tau = [2, 3, 4]

# Define two random SO3vec objects  
x = gelib.SO3vec.randn(batch, tau)

for l in range(len(tau)):
    x.parts[l].requires_grad_()

    print(x.parts[l].size())
    print(x.parts[l].is_cuda)
    tensor = x.parts[l].detach().clone()
    print(tensor[0, 0, 0, 0])

