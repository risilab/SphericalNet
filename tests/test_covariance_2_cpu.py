import torch
import torch.nn as nn

import sys
sys.path.insert(1, '../models')

import gelib_base
import gelib_torchC as gelib
from gelib_torchC import *

# Random seed
torch.manual_seed(123456789)

# Run the model
batch_size = 10
input_tau = [2, 3, 4]
inputs = gelib.SO3vec.randn(batch_size, input_tau)
outputs = gelib.CGproduct(inputs, inputs)

# Rotate the inputs and run the model again
R = gelib_base.SO3element.uniform()
inputs_rot = inputs.rotate(R)
outputs_rot = gelib.CGproduct(inputs_rot, inputs_rot)

# Check the difference
assert(len(outputs.parts) == len(outputs_rot.parts))
for l in range(len(outputs.parts)):
    first = torch.view_as_complex(outputs.rotate(R).parts[l])
    second = torch.view_as_complex(outputs_rot.parts[l])
    diff = torch.norm(first - second, p = 2)
    print(l, diff)

print('Done')
