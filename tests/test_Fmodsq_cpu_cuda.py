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
maxl = 10

# Define a random SO3vec object
x = gelib.SO3vec.Frandn(batch, maxl)

for l in range(maxl + 1):
    # x.parts[l].requires_grad_()

    x.parts[l].requires_grad_().retain_grad()

# Compute the Fmodsq
z = gelib.Fmodsq(x)

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

x_cuda = gelib.SO3vec.Frandn(batch, maxl)

for l in range(maxl + 1):
    x_cuda.parts[l] = x.parts[l].detach().to(device = 'cuda')

    x_cuda.parts[l].requires_grad_()
    
    # x_cuda.parts[l].retain_grad()

    print(x_cuda.parts[l].grad) # Should be None

z_cuda = gelib.Fmodsq(x_cuda)
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
for l in range(maxl + 1):
    print(torch.norm(z.parts[l].detach() - z_cuda.parts[l].cpu().detach()))

print('Difference in each individual tensor gradient:')
for l in range(maxl + 1):
    # print(x_cuda.parts[l].grad) # NoneType
    # print(x.parts[l].grad)
    print(torch.norm(x.parts[l].grad.detach() - x_cuda.parts[l].grad.cpu().detach()))

print('Done')
