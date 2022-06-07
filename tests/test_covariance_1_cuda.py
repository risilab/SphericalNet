import torch
import torch.nn as nn

import sys
sys.path.insert(1, '../models')

import gelib_base
import gelib_torchC as gelib
from gelib_torchC import *

# Random seed
torch.manual_seed(123456789)

# Create model
class Test_Model(torch.nn.Module):
    def __init__(self, device = 'cuda'):
        super(Test_Model, self).__init__()

        self.device = device

    def forward(self, inputs):
        outputs = gelib.CGproduct(inputs, inputs)
        return outputs

model = Test_Model()

# Device
device = 'cuda' # Turn this to 'cpu' then there is no problem

# Run the model
batch_size = 10
input_tau = [2, 3, 4]
inputs = gelib.SO3vec.randn(batch_size, input_tau)
inputs = inputs.to(device = device)
outputs = model(inputs)

# Rotate the inputs and run the model again
R = gelib_base.SO3element.uniform()
# R = R.to(device = 'cuda') # I don't understand why I can't get this one on GPU
inputs_rot = inputs.rotate(R)
outputs_rot = model(inputs_rot)

# Check the difference
assert(len(outputs.parts) == len(outputs_rot.parts))
for l in range(len(outputs.parts)):
    first = torch.view_as_complex(outputs.rotate(R).parts[l].detach().cpu())
    second = torch.view_as_complex(outputs_rot.parts[l].detach().cpu())

    print(first.is_contiguous())
    print(second.is_contiguous())
    
    print(first.size())
    print(second.size())

    diff = torch.norm(first - second, p = 2)

    print(l, diff)

print('Done')
