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
from spherical_cnn_classification import Spherical_CNN_Classification

import gzip

# S2 FFT
from s2_fft import S2_fft_real

def _parse_args():
    parser = argparse.ArgumentParser(description = 'Supervised learning')
    parser.add_argument('--dataset', '-dataset', type = str, default = 'DATASET', help = 'Dataset')
    parser.add_argument('--dir', '-dir', type = str, default = '.', help = 'Directory')
    parser.add_argument('--name', '-name', type = str, default = 'NAME', help = 'Name')
    parser.add_argument('--num_epochs', '-num_epochs', type = int, default = 2048, help = 'Number of epochs')
    parser.add_argument('--batch_size', '-batch_size', type = int, default = 20, help = 'Batch size')
    parser.add_argument('--learning_rate', '-learning_rate', type = float, default = 0.001, help = 'Initial learning rate')
    parser.add_argument('--seed', '-s', type = int, default = 123456789, help = 'Random seed')
    parser.add_argument('--num_layers', '-num_layers', type = int, default = 3, help = 'Number of layers')
    parser.add_argument('--hidden_dim', '-hidden_dim', type = int, default = 16, help = 'Hidden dimension')
    parser.add_argument('--max_l', '-max_l', type = int, default = 11, help = 'Maximum l')
    parser.add_argument('--diag_cg', '-diag_cg', type = int, default = 1, help = 'Diagonal CG product')
    parser.add_argument('--device', '-device', type = str, default = 'cpu', help = 'cuda/cpu')
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

# Hyper-parameters
dataset = args.dataset
num_epochs = args.num_epochs
learning_rate = args.learning_rate
batch_size = args.batch_size
num_layers = args.num_layers
max_l = args.max_l
hidden_tau = [args.hidden_dim for i in range(max_l + 1)]

if args.diag_cg == 1:
    diag_cg = True
else:
    diag_cg = False

# Data loader
class spherical_mnist_data(Dataset):
    def __init__(self, data_file, partition, max_l, device = 'cpu'):
        self.partition = partition
        self.images = data_file[self.partition]["images"]
        self.labels = data_file[self.partition]["labels"]
        self.num_samples = self.images.shape[0]
        self.max_l = max_l

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # S2 FFT
        example = self.images[idx]
        example = torch.from_numpy(example.astype(np.float32))
        example = example.unsqueeze(dim = 0)
        x = S2_fft_real.apply(example, self.max_l + 1)
        x = x.squeeze(dim = 1)

        # Parts
        parts = []
        start = 0
        for l in range(self.max_l + 1):
            part = x[start : start + (2 * l + 1)]
            part = part.unsqueeze(dim = 1)
            parts.append(part.detach().to(device = device))
            start += (2 * l + 1)
        assert start == x.size(0)

        # Label
        label = torch.zeros(10)
        label[self.labels[idx]] = 1
        label = label.to(device = device)

        # Sample
        sample = {
            'parts': parts,
            'label': label
        }
        return sample

# Datasets
print('Dataset:', dataset)
data_file = gzip.open(dataset, 'rb')
data_file = pickle.load(data_file)

train_dataset = spherical_mnist_data(data_file, "train", max_l, device)
test_dataset = spherical_mnist_data(data_file, "test", max_l, device)

train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle = False)

# Test the data correctness & Compute input_tau
for batch_idx, data in enumerate(train_dataloader):
    input_tau = []
    for l in range(len(data['parts'])):
        input_tau.append(data['parts'][l].size(2))
        assert(data['parts'][l].dim() == 4)
        assert(data['parts'][l].size(0) == batch_size)
        assert(data['parts'][l].size(1) == 2 * l + 1)
        assert(data['parts'][l].size(3) == 2)
        print(data['parts'][l].size())
    assert(data['label'].dim() == 2)
    assert(data['label'].size(0) == batch_size)
    assert(data['label'].size(1) == 10)
    print(data['label'].size())
    break

# Model creation
model = Spherical_CNN_Classification(10, num_layers, input_tau, hidden_tau, max_l, diag_cg = diag_cg, has_normalization = True, device = device)
optimizer = Adagrad(model.parameters(), lr = learning_rate)

# Compute accuracy
def accuracy(predict, target):
    predict = torch.argmax(predict, dim = 1)
    target = torch.argmax(target, dim = 1)
    num_samples = predict.size(0)
    acc = 0
    for i in range(num_samples):
        if predict[i] == target[i]:
            acc += 1
    acc /= num_samples
    return acc

# Train model
best_acc = 0
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
        label = data['label']

        # Construct SO(3) vector
        inputs = gelib.SO3vec(parts)

        # Run model
        outputs = model(inputs)
        optimizer.zero_grad()

        # Cross-entropy loss
        loss = F.binary_cross_entropy(outputs.view(-1), label.view(-1), reduction = 'mean')

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

    # Validation
    t = time.time()
    model.eval()

    all_predict = []
    all_target = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_dataloader):
            # Read data
            parts = data['parts']
            label = data['label']

            # Construct SO(3) vector
            inputs = gelib.SO3vec(parts)

            # Run model
            outputs = model(inputs)

            all_predict.append(outputs)
            all_target.append(label)

        # Compute accuracies
        all_predict = torch.cat(all_predict, dim = 0)
        all_target = torch.cat(all_target, dim = 0)

        acc = accuracy(all_predict, all_target)
        print('Test accuracy:', acc)
        LOG.write('Test accuracy: ' + str(acc) + '\n')

        print("Test time =", "{:.5f}".format(time.time() - t))
        LOG.write("Test time = " + "{:.5f}".format(time.time() - t) + "\n")

    if acc > best_acc:
        best_acc = acc
        print('Current best accuracy updated:', best_acc)
        LOG.write('Current best accuracy updated: ' + str(best_acc) + '\n')

        torch.save(model.state_dict(), model_name)

        print("Save the best model to " + model_name)
        LOG.write("Save the best model to " + model_name + "\n")

print('Best accuracy:', best_acc)
LOG.write('Best accuracy: ' + str(best_acc) + '\n')

print('Done')
LOG.close()

