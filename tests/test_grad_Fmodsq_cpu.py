import torch

import sys
sys.path.insert(1, '../models/')
import gelib_torchC as gelib

torch.manual_seed(123)

# ---- Fmodsq ---------------------------------------------------------------------------------------------

# Define the type 
batch = 1
maxl = 1

# Define two random SO3vec objects  
x = gelib.SO3vec.Frandn(batch, maxl)

for l in range(maxl + 1):
    x.parts[l].requires_grad_()

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

# Copying
max_diff = 0

for ell in range(maxl + 1):
    A = x.parts[ell].size(0)
    B = x.parts[ell].size(1)
    C = x.parts[ell].size(2)
    D = x.parts[ell].size(3)
    tuples = [(a, b, c, d) for a in range(A) for b in range(B) for c in range(C) for d in range(D)]

    for (a, b, c, d) in tuples:
        print('----------------------')
        print('Element at position:', a, b, c, d)

        diff = 1e9
        for delta in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
            print('Delta:', delta)

            x2 = gelib.SO3vec.Fzeros(batch, maxl)

            for l in range(maxl + 1):
                x2.parts[l] = x.parts[l].detach().clone()
            
            x2.parts[ell][a, b, c, d] += delta

            for l in range(maxl + 1):
                x2.parts[l].requires_grad_()

            z2 = gelib.Fmodsq(x2)

            # Sum all the elements
            # out2 = torch.sum(z2.parts[0]) + torch.sum(z2.parts[1]) + torch.sum(z2.parts[2])

            # Take the norm
            out2 = torch.norm(z2.parts[0]) + torch.norm(z2.parts[1]) + torch.norm(z2.parts[2])

            loss2 = torch.nn.functional.mse_loss(out2, target)
            manual_grad = (loss2 - loss) / delta

            error = abs(x.parts[ell].grad[a, b, c, d].item() - manual_grad.item())
            diff = min(diff, error)

            print('loss:', loss)
            print('loss2:', loss2)

            print('PyTorch grad:', x.parts[ell].grad[a, b, c, d].item())
            print('Manual grad:', manual_grad.item())
            print('***')

        if diff > max_diff:
            max_diff = diff

print('-----------------------')
print('Maximum difference:', max_diff)
print('Done')
