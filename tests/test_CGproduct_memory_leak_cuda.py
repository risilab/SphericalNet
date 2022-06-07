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

    # +------+
    # | CUDA |
    # +------+

    x_cuda = gelib.SO3vec.zeros(batch, tau)
    y_cuda = gelib.SO3vec.zeros(batch, tau)

    for l in range(len(tau)):
        x_cuda.parts[l] = x.parts[l].detach().to(device = 'cuda')
        y_cuda.parts[l] = y.parts[l].detach().to(device = 'cuda')

        x_cuda.parts[l].requires_grad_()
        y_cuda.parts[l].requires_grad_()
    
    start = time.time()
    z_cuda = gelib.CGproduct(x_cuda, y_cuda)
    finish = time.time()
    cuda_forward_time = (finish - start)

    print('CUDA forward time:', cuda_forward_time)

    # Check the output tensors
    out_tau = z_cuda.tau()
    L = len(out_tau)
    print('Output tau:', L, out_tau)
    print('Output tensor sizes:')
    total_elem = 0
    for l in range(L):
        print(z_cuda.parts[l].size())
        # print(z_cuda.parts[l].numel())
        total_elem += z_cuda.parts[l].numel()
    print('Total number of elements:', total_elem)

    out_cuda = torch.norm(z_cuda.parts[0]) + torch.norm(z_cuda.parts[1]) + torch.norm(z_cuda.parts[2])
    target_cuda = torch.ones(out_cuda.size()).to(device = 'cuda')
    loss_cuda = torch.nn.functional.mse_loss(out_cuda, target_cuda)

    start = time.time()
    loss_cuda.backward()
    finish = time.time()
    cuda_backward_time = (finish - start)

    print('CUDA backward time:', cuda_backward_time)

print('Done')
