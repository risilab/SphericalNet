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
        outputs = gelib.CGproduct(inputs, inputs)
        return outputs

model = Test_Model()

# Run the model
batch_size = 10
input_tau = [1, 1, 1, 1]
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

    diff = torch.norm(first - second, p = 1) / second.numel()
    print(l, diff)

    epsilon = 1e-1
    first = first.detach().cpu().numpy()
    second = second.detach().cpu().numpy()
    for column in range(2 * l + 1):
        column_failure = False
        for fragment in range(first.shape[2]):
            individual_failure = False
            for b in range(first.shape[0]):
                if abs(first[b, column, fragment] - second[b, column, fragment]) > epsilon:
                    column_failure = True
                    individual_failure = True
                    break
            if individual_failure == True:
                print('Column', column, 'and fragment', fragment, 'fails')
        if column_failure == True:
            print('Column', column, 'fails')
    print(first.shape)

print('Done')
