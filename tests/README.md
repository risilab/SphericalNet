Tests:

* ```test_covariance_1_cpu.py``` and ```test_covariance_2_cpu.py```: Tests, on CPU, for the covariance/equivariance of a simple spherical CNN model built based on the CG product from GElib.

* ```test_invariant_network.py```: Test for the rotational invariance of a spherical CNN model built based on GElib.

* ```test_grad_CGproduct_cpu.py```: Test, on CPU, for the gradient computation of ```CGproduct``` (from GElib).

* ```test_grad_Fmodsq_cpu.py```: Test, on CPU, for the gradient computation of ```Fmodsq``` (from GElib).

* ```test_grad_Fproduct_cpu.py```: Test, on CPU, for the gradient computation of ```Fproduct``` (from GElib).

* ```test_grad_network_cou.py```: Test, on CPU, for the gradient (with respect to learnable weights) of a spherical CNN model.

* ```test_CGproduct_cpu_cuda.py```: Test if CGproduct (forward & backward) is the same for either CPU or GPU.

* ```test_Fproduct_cpu_cuda.py```: Test if Fproduct (forward & backward) is the same for either CPU or GPU.

* ```test_Fmodsq_cpu_cuda.py```: Test if Fmodsq (forward & backward) is the same for either CPU or GPU.

* ```test_CGproduct_performance.py```: Performance test for CGproduct on CPU vs GPU.

* ```test_Fproduct_performance.py```: Performance test for Fproduct on CPU vs GPU.

* ```test_Fmodsq_performance.py```: Performance test for Fmodsq on CPU vs GPU.
