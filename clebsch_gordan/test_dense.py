import torch
import cg_cpp

# Initialize the CG table in dense format in CPU
L = 3
dense_tensor = cg_cpp.dense_tensor(L)

print('L =', L)
print('Dense tensor size:', dense_tensor.size())

l = 2;
l1 = 1;
l2 = 1;
m = -1; # Ranging from -l to l
m1 = 1; # Ranging from -l1 to l1
m2 = 0; # Ranging from -l2 to l2

value = dense_tensor[l, l1, l2, m + l, m1 + l1, m2 + l2];
print(value)

