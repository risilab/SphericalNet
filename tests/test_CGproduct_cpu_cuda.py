import torch

import sys
sys.path.insert(1, '../models/')
import gelib_torchC as gelib

torch.manual_seed(123)

# +-----+
# | CPU |
# +-----+

# Define the type 
batch = 10
tau = [2, 3, 4]

# Define two random SO3vec objects  
x = gelib.SO3vec.randn(batch, tau)
y = gelib.SO3vec.randn(batch, tau)

for l in range(len(tau)):
    # x.parts[l].requires_grad_()
    # y.parts[l].requires_grad_()

    x.parts[l].requires_grad_().retain_grad()
    y.parts[l].requires_grad_().retain_grad()

# Compute the CG-product
z = gelib.CGproduct(x, y)

# Sum all elements
# out=torch.sum(z.parts[0])+torch.sum(z.parts[1])+torch.sum(z.parts[2])

# Take the norm
out = torch.norm(z.parts[0]) + torch.norm(z.parts[1]) + torch.norm(z.parts[2])

target = torch.ones(out.size())
loss = torch.nn.functional.mse_loss(out,target)

# Backward
loss.backward()

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

    print(x_cuda.parts[l].grad) # Should be None
    print(y_cuda.parts[l].grad) # Should be None

z_cuda = gelib.CGproduct(x_cuda, y_cuda)
out_cuda = torch.norm(z_cuda.parts[0]) + torch.norm(z_cuda.parts[1]) + torch.norm(z_cuda.parts[2])
target_cuda = torch.ones(out_cuda.size()).to(device = 'cuda')
loss_cuda = torch.nn.functional.mse_loss(out_cuda, target_cuda)
loss_cuda.backward()

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

print('Done')
