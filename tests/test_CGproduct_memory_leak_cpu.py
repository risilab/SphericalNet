import torch
import time

import sys
sys.path.insert(1, '../models/')
import gelib_torchC as gelib

torch.manual_seed(123)

# +-----+
# | CPU |
# +-----+

# Define the type 
batch = 10
maxl = 8
tau = [32 for l in range(maxl + 1)]

for test_idx in range(1000):
    # Define two random SO3vec objects  
    x = gelib.SO3vec.randn(batch, tau)
    y = gelib.SO3vec.randn(batch, tau)

    for l in range(len(tau)):
        x.parts[l].requires_grad_().retain_grad()
        y.parts[l].requires_grad_().retain_grad()

    # Compute the CG-product
    start = time.time()
    z = gelib.CGproduct(x, y)
    finish = time.time()
    cpu_forward_time = (finish - start)

    # Sum all elements
    # out=torch.sum(z.parts[0])+torch.sum(z.parts[1])+torch.sum(z.parts[2])

    # Take the norm
    out = torch.norm(z.parts[0]) + torch.norm(z.parts[1]) + torch.norm(z.parts[2])

    target = torch.ones(out.size())
    loss = torch.nn.functional.mse_loss(out,target)

    # Backward
    start = time.time()
    loss.backward()
    finish = time.time()
    cpu_backward_time = (finish - start)

    print('---- Time ----')
    print('CPU forward time:', cpu_forward_time)
    print('CPU backward time:', cpu_backward_time)

print('Done')
