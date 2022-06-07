import torch
import time

import sys
sys.path.insert(1, '../models/')
import gelib_torchC as gelib

torch.manual_seed(123)

# Define the type 
batch = 20
maxl = 3

# Initialize a filter 
f = gelib.SO3vec.Frandn(batch, maxl)

print("\nFilter size:")
for l in range(len(f.parts)):
    print(f.parts[l].size())

# Fmodsq of the filter (we limit with the same maxl)
f_modsq = gelib.Fmodsq(f, maxl)

print("\nfmodsq of the filter size:")
for l in range(len(f_modsq.parts)):
    print(f_modsq.parts[l].size())

# Heat (some maxl random numbers here)
heat = [2, 1, 3, 3]

# Penalizer
p_parts = []
for l in range(maxl + 1):
    # Identity matrix
    part = torch.cat([torch.eye(2 * l + 1).unsqueeze(dim = 0) for b in range(batch)], dim = 0)
    
    # Padd with a zero complex part
    part = torch.cat([part.unsqueeze(dim = 3), torch.zeros(part.size()).unsqueeze(dim = 3)], dim = 3)

    # Multiply with the heat
    part = heat[l] * part

    # Add to the list
    p_parts.append(part)

p = gelib.SO3vec(p_parts)

print("\nPenalizer size:")
for l in range(len(p.parts)):
    print(p.parts[l].size())

# Penalizing
g = gelib.Fproduct(f_modsq, p, maxl)

# Suppose we have an SO3vec as the input signal
x = gelib.SO3vec.randn(batch, [10, 20, 30, 40])

print("\nInput signal size:")
for l in range(len(x.parts)):
    print(x.parts[l].size())

# Apply the regularized filter to x
y_parts = []
for l in range(maxl + 1):
    part = torch.einsum('bijv,bjkv->bikv', g.parts[l], x.parts[l])
    y_parts.append(part)

y = gelib.SO3vec(y_parts) # This the result!

print("\nFinal result size:")
for l in range(len(y.parts)):
    print(y.parts[l].size())

print('Done')
