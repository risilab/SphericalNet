import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import Adam, Adagrad
from torch.utils.data import Dataset, DataLoader

import numpy as np
import json
import os
import time
import argparse
import pickle

import sys
sys.path.insert(1, '../../models')

import gelib_torchC as gelib
from gelib_torchC import *
from spherical_cnn_regression import Spherical_CNN_Regression

def _parse_args():
    parser = argparse.ArgumentParser(description = 'Supervised learning')
    parser.add_argument('--dir', '-dir', type = str, default = '.', help = 'Directory')
    parser.add_argument('--name', '-name', type = str, default = 'NAME', help = 'Name')
    parser.add_argument('--num_epochs', '-num_epochs', type = int, default = 2048, help = 'Number of epochs')
    parser.add_argument('--batch_size', '-batch_size', type = int, default = 20, help = 'Batch size')
    parser.add_argument('--learning_rate', '-learning_rate', type = float, default = 0.001, help = 'Initial learning rate')
    parser.add_argument('--seed', '-s', type = int, default = 123456789, help = 'Random seed')
    parser.add_argument('--num_layers', '-num_layers', type = int, default = 3, help = 'Number of layers')
    parser.add_argument('--hidden_dim', '-hidden_dim', type = int, default = 16, help = 'Hidden dimension')
    parser.add_argument('--max_l', '-max_l', type = int, default = 2, help = 'Maximum l')
    parser.add_argument('--diag_cg', '-diag_cg', type = int, default = 1, help = 'Diagonal CG product')
    parser.add_argument('--device', '-device', type = str, default = 'cuda', help = 'cuda/cpu')
    args = parser.parse_args()
    return args

args = _parse_args()
log_name = args.dir + "/" + args.name + ".log"
model_name = args.dir + "/" + args.name + ".model"
LOG = open(log_name, "w")

# Fix CPU torch random seed
torch.manual_seed(args.seed)

# Fix GPU torch random seed
torch.cuda.manual_seed(args.seed)

# Fix the Numpy random seed
np.random.seed(args.seed)

# Train on CPU (hide GPU) due to memory constraints
# os.environ['CUDA_VISIBLE_DEVICES'] = ""
device = args.device

print('Log will be saved to:', log_name)
print('Model will be saved to:', model_name)
print('Run on device:', device)

# Synthetic dataset
class synthetic_dataset(Dataset):
    def __init__(self, input_tau, num_samples):
        self.input_tau = input_tau
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Parts
        parts = [torch.randn(2 * l + 1, self.input_tau[l], 2) for l in range(len(self.input_tau))]

        # Synthetic
        label = 0
        for l in range(len(self.input_tau)):
            label += torch.norm(parts[l], p = 2).item()

        # Sample
        sample = {
            'parts': parts,
            'label': label
        }
        return sample

# Hyper-parameters
num_epochs = args.num_epochs
learning_rate = args.learning_rate
batch_size = args.batch_size
num_layers = args.num_layers
max_l = args.max_l
input_tau = [8 for l in range(max_l + 1)]
hidden_tau = [args.hidden_dim for l in range(max_l + 1)]

if args.diag_cg == 1:
    diag_cg = True
else:
    diag_cg = False


# Datasets
train_dataset = synthetic_dataset(input_tau, num_samples = 10000)
test_dataset = synthetic_dataset(input_tau, num_samples = 1000)

train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle = False)

'''
# Test the synthetic data correctness
for batch_idx, data in enumerate(train_dataloader):
    assert(len(data['parts']) == len(input_tau))
    for l in range(len(data['parts'])):
        assert(data['parts'][l].dim() == 4)
        assert(data['parts'][l].size(0) == batch_size)
        assert(data['parts'][l].size(1) == 2 * l + 1)
        assert(data['parts'][l].size(3) == 2)
        print(data['parts'][l].size())
    assert(data['label'].dim() == 1)
    assert(data['label'].size(0) == batch_size)
    print(data['label'].size())
    break
'''

# Model creation
model = Spherical_CNN_Regression(num_layers, input_tau, hidden_tau, max_l, diag_cg = diag_cg, has_normalization = True, device = device)
optimizer = Adagrad(model.parameters(), lr = learning_rate)

# train model
best_mae = 1e9
for epoch in range(num_epochs):
    print('--------------------------------------')
    print('Epoch', epoch)
    LOG.write('--------------------------------------\n')
    LOG.write('Epoch ' + str(epoch) + '\n')

    # Training
    t = time.time()
    total_loss = 0.0
    nBatch = 0
    for batch_idx, data in enumerate(train_dataloader):
        # Read data
        parts = data['parts']
        label = data['label'].to(device = device)

        # Construct SO(3) vector
        inputs = gelib.SO3vec(parts)

        # Run model
        outputs = torch.abs(torch.view_as_complex(model(inputs)))
        optimizer.zero_grad()

        # Mean squared error loss
        # loss = F.mse_loss(outputs.view(-1), label.view(-1), reduction = 'mean')

        # L1 loss
        loss = F.l1_loss(outputs.view(-1), label.view(-1), reduction = 'mean')

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        nBatch += 1
        if batch_idx % 100 == 0:
            print('Batch', batch_idx, '/', len(train_dataloader),': Loss =', loss.item())
            LOG.write('Batch ' + str(batch_idx) + '/' + str(len(train_dataloader)) + ': Loss = ' + str(loss.item()) + '\n')

    avg_loss = total_loss / nBatch
    print('Train average loss:', avg_loss)
    LOG.write('Train average loss: ' + str(avg_loss) + '\n')
    print("Train time =", "{:.5f}".format(time.time() - t))
    LOG.write("Train time = " + "{:.5f}".format(time.time() - t) + "\n")

    # Testing
    t = time.time()
    model.eval()
    with torch.no_grad():
        sum_error = 0.0
        num_samples = 0
        for batch_idx, data in enumerate(test_dataloader):
            # Read data
            parts = data['parts']
            label = data['label'].to(device = device)

            # Construct SO(3) vector
            inputs = gelib.SO3vec(parts)

            # Run model
            outputs = torch.abs(torch.view_as_complex(model(inputs)))

            # Error
            sum_error += torch.sum(torch.abs(outputs.view(-1) - label.view(-1))).detach().cpu().numpy()
            num_samples += outputs.size(0)
        mae = sum_error / num_samples

        print('Test MAE:', mae)
        LOG.write('Test MAE: ' + str(mae) + '\n')
        print("Test time =", "{:.5f}".format(time.time() - t))
        LOG.write("Test time = " + "{:.5f}".format(time.time() - t) + "\n")

    if mae < best_mae:
        best_mae = mae
        print('Current best MAE updated:', best_mae)
        LOG.write('Current best MAE updated: ' + str(best_mae) + '\n')

        torch.save(model.state_dict(), model_name)

        print("Save the best model to " + model_name)
        LOG.write("Save the best model to " + model_name + "\n")

print('Best MAE:', best_mae)
LOG.write('Best MAE: ' + str(best_mae) + '\n')

print('Done')
LOG.close()

