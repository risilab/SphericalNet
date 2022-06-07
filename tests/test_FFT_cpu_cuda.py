import torch
import gelib

torch.manual_seed(123)

# CPU
v_cpu = gelib.SO3vec.Frandn(1, 6, 0)        # define a random SO3-vector
v_cpu.parts[2].requires_grad_()           # make sure that we can backprop

r_cpu = v_cpu.iFFT(20)                          # compute an inverse Fourier transform with resolution 20

r_cpu.backward(r_cpu)                         # backprop something

grad_cpu = v_cpu.parts[2].grad / v_cpu.parts[2]    # should be more or less constant

# CUDA
v_cuda = v_cpu.to(device = 'cuda')
v_cuda.parts[2].requires_grad_()

r_cuda = v_cuda.iFFT(20)

r_cuda.backward(r_cuda)

grad_cuda = v_cuda.parts[2].grad / v_cuda.parts[2]

