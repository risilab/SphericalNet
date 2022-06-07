import torch
import time

import sys
sys.path.insert(1, '../models/')
import gelib_torchC as gelib

torch.manual_seed(123)

# Factors
batch_list = [10, 50]
tau_list = [32, 64]
maxl_list = [4, 8, 16, 32, 48]

# Measuring times
def measure_times(batch, tau, maxl):
    # +-----+
    # | CPU |
    # +-----+

    # Define two random SO3vec objects  
    x = gelib.SO3vec.randn(batch, tau)
    y = gelib.SO3vec.randn(batch, tau)

    for l in range(len(tau)):
        # x.parts[l].requires_grad_()
        # y.parts[l].requires_grad_()

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

    # +------+
    # | CUDA |
    # +------+

    # Should not use this initialization
    # x_cuda = x.to(device = 'cuda')
    # y_cuda = y.to(device = 'cuda')

    x_cuda = gelib.SO3vec.zeros(batch, tau)
    y_cuda = gelib.SO3vec.zeros(batch, tau)

    for l in range(len(tau)):
        x_cuda.parts[l] = x.parts[l].detach().to(device = 'cuda')
        y_cuda.parts[l] = y.parts[l].detach().to(device = 'cuda')

        x_cuda.parts[l].requires_grad_()
        y_cuda.parts[l].requires_grad_()
    
        # x_cuda.parts[l].retain_grad()
        # y_cuda.parts[l].retain_grad()

        # print(x_cuda.parts[l].grad) # Should be None
        # print(y_cuda.parts[l].grad) # Should be None

    start = time.time()
    z_cuda = gelib.CGproduct(x_cuda, y_cuda)
    finish = time.time()
    cuda_forward_time = (finish - start)

    out_cuda = torch.norm(z_cuda.parts[0]) + torch.norm(z_cuda.parts[1]) + torch.norm(z_cuda.parts[2])
    target_cuda = torch.ones(out_cuda.size()).to(device = 'cuda')
    loss_cuda = torch.nn.functional.mse_loss(out_cuda, target_cuda)

    start = time.time()
    loss_cuda.backward()
    finish = time.time()
    cuda_backward_time = (finish - start)

    # +----------------+
    # | Check accuracy |
    # +----------------+

    print('Difference in the loss (should be 0):')
    print(torch.norm(loss.detach() - loss_cuda.cpu().detach()))

    print('Difference in the output (should be 0):')
    print(torch.norm(out.detach() - out_cuda.cpu().detach()))

    print('Difference in each individual tensor:')
    for l in range(len(tau)):
        print(torch.norm(z.parts[l].detach() - z_cuda.parts[l].cpu().detach()))

    print('Difference in each individual tensor gradient:')
    for l in range(len(tau)):
        # print(x_cuda.parts[l].grad) # NoneType
        # print(x.parts[l].grad)
        print(torch.norm(x.parts[l].grad.detach() - x_cuda.parts[l].grad.cpu().detach()))

    for l in range(len(tau)):
        # print(y_cuda.parts[l].grad)
        print(torch.norm(y.parts[l].grad.detach() - y_cuda.parts[l].grad.cpu().detach()))

    print('---- Time ----')
    print('CPU forward time:', cpu_forward_time)
    print('CUDA forward time:', cuda_forward_time)
    print('CPU backward time:', cpu_backward_time)
    print('CUDA backward time:', cuda_backward_time)

    return cpu_forward_time, cuda_forward_time, cpu_backward_time, cuda_backward_time

# +--------------+
# | Main program |
# +--------------+

tables = []
for b in range(len(batch_list)):
    for t in range(len(tau_list)):
        for m in range(len(maxl_list)):
            batch = batch_list[b]
            maxl = maxl_list[m]
            tau = [tau_list[t] for i in range(maxl + 1)]
            print('------------------------------------------------')
            print('batch =', batch, ', tau =', tau_list[t], ', maxl =', maxl, ':')

            cpu_forward_time, cuda_forward_time, cpu_backward_time, cuda_backward_time = measure_times(batch, tau, maxl)
            tables.append(
                (batch, tau_list[t], maxl, cpu_forward_time, cuda_forward_time, cpu_backward_time, cuda_backward_time)
            )

# Summary
print('---- Summary ----')
for element in tables:
    batch, tau, maxl, cpu_forward_time, cuda_forward_time, cpu_backward_time, cuda_backward_time = element
    print('batch =', batch, ', tau =', tau, ', maxl =', maxl, ':')
    print('CPU forward time:', cpu_forward_time)
    print('CUDA forward time:', cuda_forward_time)
    print('CPU backward time:', cpu_backward_time)
    print('CUDA backward time:', cuda_backward_time)
    print('')

print('Done')
