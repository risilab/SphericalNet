import torch
import torch.nn as nn

import sys
sys.path.insert(1, '../models')
import gelib_base
import gelib_torchC as gelib
from gelib_torchC import *
from spherical_cnn_regression import Spherical_CNN_Regression

# Random seed
torch.manual_seed(123456789)

# Device
# device = 'cpu'
device = 'cuda'

# Use DiagCGproduct
diag_cg = True

# Create model
batch_size = 10
num_layers = 3
input_tau = [2, 2, 2]
hidden_tau = [4, 4, 4]

model = Spherical_CNN_Regression(num_layers, input_tau, hidden_tau, maxl = len(hidden_tau), diag_cg = diag_cg, has_normalization = True, device = device)

# Inputs and Rotated inputs
inputs = gelib.SO3vec.randn(batch_size, input_tau)
R = gelib_base.SO3element.uniform()
inputs_rot = inputs.rotate(R)

# Convert data to the device
inputs = inputs.to(device = device)
inputs_rot = inputs_rot.to(device = device)

# Run the first one
outputs = model(inputs).detach().cpu()
print('Done the first one')

# Rotate the inputs and run the model again
outputs_rot = model(inputs_rot).detach().cpu()
print('Done the second one')

# Check the difference
first = torch.view_as_complex(outputs)
second = torch.view_as_complex(outputs_rot)
diff = torch.abs(first - second)

print('Scalar outputs on the original inputs:')
print(first)
print('Scalar outputs on the rotated inputs:')
print(second)
print('Difference:')
print(diff)

diff = torch.norm(torch.abs(outputs - outputs_rot), p = 2)
print('Total difference:', diff)

print('Done')
