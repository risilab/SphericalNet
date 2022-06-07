import torch
import cg_cpp

# Initialize the CG table in sparse format in CPU
L = 10
l_indices, l1_indices, l2_indices, m_indices, m1_indices, m2_indices, values = cg_cpp.sparse_tensor(L)

# Indices for the sparse tensor
indices = torch.cat([
    l_indices.unsqueeze(dim = 0), 
    l1_indices.unsqueeze(dim = 0),
    l2_indices.unsqueeze(dim = 0),
    m_indices.unsqueeze(dim = 0),
    m1_indices.unsqueeze(dim = 0),
    m2_indices.unsqueeze(dim = 0)], dim = 0)

# The sparse tensor
sparse_tensor = torch.sparse_coo_tensor(indices, values, (L + 1, L + 1, L + 1, 2 * L + 1, 2 * L + 1, 2 * L + 1))

# Test the sparse tensor
l = 2;
l1 = 1;
l2 = 1;
m = -1; # Ranging from -l to l
m1 = 1; # Ranging from -l1 to l1
m2 = 0; # Ranging from -l2 to l2

value = sparse_tensor[l, l1, l2, m + l, m1 + l1, m2 + l2];
print('Value in the sparse tensor:', value)

'''
# We can convert it to the dense tensor also
dense_tensor = sparse_tensor.to_dense()
value = dense_tensor[l, l1, l2, m + l, m1 + l1, m2 + l2];
print('Value in the dense tensor (must be the same):', value)
'''

# Number of non-zeros
num_nonzeros = l_indices.size(0)
num_total = ((L + 1)**3) * ((2 * L + 1)**3)
print('Number of non-zeros:', num_nonzeros)
print('Number of (theoretical) elements:', num_total)
print('Sparsity (percent):', num_nonzeros / num_total * 100)
print('Done')
