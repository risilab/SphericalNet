import torch
import numpy as np
import gzip
import pickle

from s2_fft import S2_fft_real

f = gzip.open('s2_mnist.gz', 'rb')
dataset = pickle.load(f)

maxl = 11
file_prefix = 'spherical_mnist'

train_images = dataset['train']['images']
train_labels = dataset['train']['labels']
test_images = dataset['test']['images']
test_labels = dataset['test']['labels']

def to_SO3vecs(images, labels, maxl):
    SO3vecs = []
    binary_labels = []
    num_samples = images.shape[0]
    for idx in range(num_samples):
        sample = images[idx]
        sample = torch.from_numpy(sample.astype(np.float32))
        sample = sample.unsqueeze(dim = 0).unsqueeze(dim = 1)
        x = S2_fft_real.apply(sample, maxl + 1)

        parts = []
        start = 0
        for l in range(maxl + 1):
            part = x[start : start + (2 * l + 1)]
            part = torch.transpose(part, 0, 1)
            part = part.squeeze(dim = 0)
            parts.append(part.detach().cpu().numpy())
            start += (2 * l + 1)
        assert start == x.size(0)

        SO3vecs.append(parts)
        
        label = labels[idx]
        binary_label = np.zeros((10))
        binary_label[label] = 1
        binary_labels.append(binary_label)

        if (idx + 1) % 100 == 0:
            print('Done', idx + 1)
    return SO3vecs, binary_labels

def save_to_file(data, file_name):
    f = open(file_name, 'wb')
    pickle.dump(data, f)
    f.close()

train_SO3vecs, train_labels = to_SO3vecs(train_images, train_labels, maxl)
print('Done S2-FFT for the train set')

test_SO3vecs, test_labels = to_SO3vecs(test_images, test_labels, maxl)
print('Done S2-FFT for the test set')

save_to_file((train_SO3vecs, train_labels, test_SO3vecs, test_labels), file_prefix + '_maxl_' + str(maxl) + '.pkl')
print('Save to file')
print('Done')

