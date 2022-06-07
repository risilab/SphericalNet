import torch
import numpy as np
import gzip
import pickle

from s2_fft import S2_fft_real

f = gzip.open('s2_mnist.gz', 'rb')
dataset = pickle.load(f)

example = dataset['train']['images'][0:10]
example = torch.from_numpy(example.astype(np.float32))
print(example.size())
example = example.unsqueeze(dim = 1)

L = 13
print(example.size())
x = S2_fft_real.apply(example, L + 1)

print(x.size())
