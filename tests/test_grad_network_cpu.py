import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Adagrad
from torch import optim
from torch.utils.data import DataLoader

import sys
sys.path.insert(1, '../models')

import gelib_torchC as gelib
from gelib_torchC import *
from spherical_cnn_regression import Spherical_CNN_Regression

# Random seed
torch.manual_seed(123456789)

# Device
device = 'cpu'
# device = 'cuda'

# DiagCGproduct
diag_cg = False

# Create model
batch_size = 10
num_layers = 3
input_tau = [8, 8]
hidden_tau = [16, 16, 16]

model = Spherical_CNN_Regression(num_layers, input_tau, hidden_tau, maxl = len(hidden_tau), diag_cg = diag_cg, has_normalization = True, device = device)

# Fixed target
targets = torch.ones(batch_size, 2)

# All parameters
optimizer = Adagrad(model.parameters(), lr = 1e-3)

# Run the model
inputs = gelib.SO3vec.randn(batch_size, input_tau)
outputs = model(inputs)

# Check learnable parameters
print('-------------------')
count = 0
for param in model.parameters():
    if param.requires_grad:
        print(param.data.size())
        count += 1
assert(count == model.num_learnable_params)
print('-------------------')

# Gradient tests
delta = 1e-3
for param in model.parameters():
    if param.requires_grad:
        assert(param.dim() == 3)
        num_rows = param.size(0)
        num_cols = param.size(1)
        assert(param.size(2) == 2)
        for row in range(num_rows):
            for col in range(num_cols):
                for i in range(2):
                    # PyTorch gradient
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    
                    # torch.norm
                    # loss = torch.norm(outputs - targets, p = 1)
                    
                    # torch.nn.functional.mse_loss
                    loss = torch.nn.functional.mse_loss(outputs, targets)

                    loss.backward()
                    pytorch_grad = param.grad.data.detach().clone()[row, col, i].item()
                    print('PyTorch grad:', pytorch_grad)

                    # Manual gradient
                    param.data[row, col, i] += delta
                    outputs_2 = model(inputs)
                    
                    # torch.norm
                    # loss_2 = torch.norm(outputs_2 - targets, p = 1)
                    
                    # torch.nn.functional.mse_loss
                    loss_2 = torch.nn.functional.mse_loss(outputs_2, targets)

                    manual_grad = (loss_2.item() - loss.item()) / delta
                    param.data[row, col, i] -= delta
                    print('Manual grad:', manual_grad)
                    print('Difference:', abs(pytorch_grad - manual_grad))
                    print('-----------------------')

print('Done')
