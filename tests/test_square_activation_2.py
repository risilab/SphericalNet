import torch
import time

import sys
sys.path.insert(1, '../models/')
import gelib_torchC as gelib

torch.manual_seed(123)

# Tensor normalization
def tensor_normalization(inputs):
    outputs = []
    for i in range(len(inputs)):
        tensor = torch.view_as_complex(inputs[i])
        norm = torch.einsum('bij,bij->b', tensor, tensor)
        norm = 1.0 / norm
        out = torch.view_as_real(torch.einsum('bij,b->bij', tensor, norm))
        outputs.append(out)
    return outputs

# Define the type 
batch = 20
input_tau = [2, 3, 4, 5, 6, 7]
input_maxl = len(input_tau) - 1
hidden_maxl = 3

# Input signal
input_signal = gelib.SO3vec.randn(batch, input_tau)

print('--------------------------------')
print('input_signal')
for l in range(len(input_signal.parts)):
    print(input_signal.parts[l].size())

# First layer filter
first_filter_parts = []
for l in range(input_maxl + 1):
    part = torch.randn(batch, 2 * l + 1, input_tau[l], 2)
    first_filter_parts.append(part)
first_filter = gelib.SO3vec(first_filter_parts)

# First layer outcome
first_layer_parts = []
for l in range(input_maxl + 1):
    part_1 = torch.view_as_complex(input_signal.parts[l])
    part_2 = torch.view_as_complex(first_filter.parts[l])

    part_3 = torch.einsum('bmi,bni->bmni', part_1, part_2)
    
    linear_map = torch.view_as_complex(torch.randn(input_tau[l], 2))
    part_4 = torch.einsum('bmni,i->bmn', part_3, linear_map)
    part_4 = torch.view_as_real(part_4)

    first_layer_parts.append(part_4)

first_layer = gelib.SO3vec(tensor_normalization(first_layer_parts))

print('--------------------------------')
print('first_layer')
for l in range(len(first_layer.parts)):
    print(first_layer.parts[l].size())

# Second layer
second_filter_parts = []
for l in range(input_maxl + 1):
    part = torch.randn(batch, 2 * l + 1, 2 * l + 1, 2)
    second_filter_parts.append(part)
second_filter = gelib.SO3vec(second_filter_parts)

# Second layer outcome
second_layer = gelib.Fproduct(first_layer, second_filter, hidden_maxl)
second_layer = gelib.Fmodsq(second_layer, hidden_maxl)
second_layer = gelib.SO3vec(tensor_normalization(second_layer.parts))

print('--------------------------------')
print('second_layer')
for l in range(len(second_layer.parts)):
    print(second_layer.parts[l].size())

# Third layer
third_filter_parts = []
for l in range(hidden_maxl + 1):
    part = torch.randn(batch, 2 * l + 1, 2 * l + 1, 2)
    third_filter_parts.append(part)
third_filter = gelib.SO3vec(third_filter_parts)

# Third layer outcome
third_layer = gelib.Fproduct(second_layer, third_filter, hidden_maxl)
third_layer = gelib.Fmodsq(third_layer, hidden_maxl)
third_layer = gelib.SO3vec(tensor_normalization(third_layer.parts))

print('--------------------------------')
print('third_layer')
for l in range(len(third_layer.parts)):
    print(third_layer.parts[l].size())

print(third_layer)
print('Done')
