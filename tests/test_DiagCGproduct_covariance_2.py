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
    def __init__(self, diag = True):
        super(Test_Model, self).__init__()
        self.diag = diag

    def forward(self, inputs):
        if self.diag == True:
            outputs = gelib.DiagCGproduct(inputs, inputs)
        else:
            outputs = gelib.CGproduct(inputs, inputs)
        return outputs

model_diag = Test_Model(diag = True)
model_normal = Test_Model(diag = False)

# Run the model
batch_size = 10
input_tau = [1, 1, 1, 1]
inputs = gelib.SO3vec.randn(batch_size, input_tau)

# DiagCGproduct
diag_outputs = model_diag(inputs)

# CGproduct
normal_outputs = model_normal(inputs)

# Check the difference
diag_vs_normal = []
assert(len(diag_outputs.parts) == len(normal_outputs.parts))
for l in range(len(diag_outputs.parts)):
    first = torch.view_as_complex(diag_outputs.parts[l])
    second = torch.view_as_complex(normal_outputs.parts[l])

    total = torch.norm(first - second, p = 1)
    average = total / second.numel()

    outcome = ('l = ' + str(l), 'total = ' + str(total.item()), 'average = ' + str(average.item()))
    diag_vs_normal.append(outcome)

# Rotate the inputs and run the model again
R = gelib_base.SO3element.uniform()
inputs_rot = inputs.rotate(R)

diag_outputs_rot = model_diag(inputs_rot)
normal_outputs_rot = model_normal(inputs_rot)

# Check the difference
diag_covariance = []
assert(len(diag_outputs.parts) == len(diag_outputs_rot.parts))
for l in range(len(diag_outputs.parts)):
    first = torch.view_as_complex(diag_outputs.rotate(R).parts[l])
    second = torch.view_as_complex(diag_outputs_rot.parts[l])

    total = torch.norm(first - second, p = 1)
    average = total / second.numel()

    outcome = ('l = ' + str(l), 'total = ' + str(total.item()), 'average = ' + str(average.item()))
    diag_covariance.append(outcome)

normal_covariance = []
assert(len(normal_outputs.parts) == len(normal_outputs_rot.parts))
for l in range(len(normal_outputs.parts)):
    first = torch.view_as_complex(normal_outputs.rotate(R).parts[l])
    second = torch.view_as_complex(normal_outputs_rot.parts[l])

    total = torch.norm(first - second, p = 1)
    average = total / second.numel()

    outcome = ('l = ' + str(l), 'total = ' + str(total.item()), 'average = ' + str(average.item()))
    normal_covariance.append(outcome)

print('Diag vs Normal -----------------')
print(diag_vs_normal)

print('Diag covariance --------------------')
print(diag_covariance)

print('Normal covariance --------------------')
print(normal_covariance)

print('Done')
