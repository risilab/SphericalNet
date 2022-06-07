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
    def __init__(self):
        super(Test_Model, self).__init__()

    def forward(self, inputs):
        outputs = gelib.DiagCGproduct(inputs, inputs)
        return outputs

model = Test_Model()

# Run the model
batch_size = 10
input_tau = [4, 4, 4]
inputs = gelib.SO3vec.randn(batch_size, input_tau)
outputs = model(inputs)

# Rotate the inputs and run the model again
R = gelib_base.SO3element.uniform()
inputs_rot = inputs.rotate(R)
outputs_rot = model(inputs_rot)

# Check the difference
assert(len(outputs.parts) == len(outputs_rot.parts))
for l in range(len(outputs.parts)):
    first = torch.view_as_complex(outputs.rotate(R).parts[l])
    second = torch.view_as_complex(outputs_rot.parts[l])

    # print(first.is_contiguous())
    # print(second.is_contiguous())
    
    # print(first.size())
    # print(second.size())

    diff = torch.norm(first - second, p = 2)

    print(l, diff)

print('Done')
