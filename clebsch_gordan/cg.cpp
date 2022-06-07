#include <iostream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <thread>
#include <assert.h>

#include <torch/torch.h>

using namespace std;

// 2 indices
static int Index(
	const int x1, 
	const int x2, 
	const int X2) 
{
	assert(x2 < X2);
	return x1 * X2 + x2;
}

// 3 indices 
static int Index(
	const int x1, 
	const int x2, 
	const int x3, 
	const int X2, 
	const int X3) 
{
	assert(x2 < X2);
	assert(x3 < X3);
	return (x1 * X2 + x2) * X3 + x3;
}

// 4 indices
static int Index(
	const int x1, 
	const int x2, 
	const int x3, 
	const int x4, 
	const int X2, 
	const int X3, 
	const int X4) 
{
	assert(x2 < X2);
	assert(x3 < X3);
	assert(x4 < X4);
	return ((x1 * X2 + x2) * X3 + x3) * X4 + x4;
}

// 5 indices
static int Index(
        const int x1,
        const int x2,
        const int x3,
        const int x4,
	const int x5,	
        const int X2, 
        const int X3,
        const int X4,
	const int X5)
{
        assert(x2 < X2);
        assert(x3 < X3);
        assert(x4 < X4);
	assert(x5 < X5);
        return (((x1 * X2 + x2) * X3 + x3) * X4 + x4) * X5 + x5;
}

// 6 indices
static int Index(
        const int x1,
        const int x2,
        const int x3,
        const int x4,
        const int x5,
	const int x6,
        const int X2,
        const int X3,
        const int X4,
        const int X5,
	const int X6)
{
        assert(x2 < X2);
        assert(x3 < X3);
        assert(x4 < X4);
        assert(x5 < X5);
	assert(x6 < X6);
        return ((((x1 * X2 + x2) * X3 + x3) * X4 + x4) * X5 + x5) * X6 + x6;
}

// +---------------------------------+
// | For CG coefficients computation |
// +---------------------------------+

int fact(int n) {
	int result = 1;
	for (int i = 2; i <= n; ++i) {
		result *= i;
	}
	return result;
}

double logfact(int n) {
	double result = 0;
	for (int i = 2; i <= n; ++i) {
		result += log((double) i);
	}
	return result;
}

double plusminus(int k) {
	if (k % 2 == 1) {
		return -1;
	}
	return 1;
}

// m ranges in [-l, l]
// m1 ranges in [-l1, l1]
// m2 ranges in [-l2, l2]
double slowCG(int l, int l1, int l2, int m, int m1, int m2, double *LogFact) {
	int m3 = -m;
	int t1 = l2 - m1 - l;
	int t2 = l1 + m2 - l;
	int t3 = l1 + l2 - l;
	int t4 = l1 - m1;
	int t5 = l2 + m2;

	int tmin = max(0, max(t1, t2));
	int tmax = min(t3, min(t4, t5));

	double wigner = 0;

	double logA = (LogFact[l1 + l2 - l] + LogFact[l1 - l2 + l] + LogFact[-l1 + l2 + l] - LogFact[l1 + l2 + l + 1]) / 2;
	logA += (LogFact[l1 + m1] + LogFact[l1 - m1] + LogFact[l2 + m2] + LogFact[l2 - m2] + LogFact[l + m3] + LogFact[l - m3]) / 2;

	for (int t = tmin; t <= tmax; ++t) {
		double logB = LogFact[t] + LogFact[t - t1] + LogFact[t - t2] + LogFact[t3 - t] + LogFact[t4 - t] + LogFact[t5 - t];
		wigner += plusminus(t) * exp(logA - logB);
	}

	return plusminus(l1 - l2 - m3) * plusminus(l1 - l2 + m) * sqrt((double)(2 * l + 1)) * wigner;
}

// +--------------+
// | Dense Tensor |
// +--------------+

at::Tensor dense_tensor(const int L) {
	// Memory allocation
	at::Tensor tensor = torch::zeros({L + 1, L + 1, L + 1, 2 * L + 1, 2 * L + 1, 2 * L + 1});
	float *flat = reinterpret_cast<float*>(tensor.data<float>());

	// Precomputatin for log of factorials	
	const int N = 3 * L + 1;
	double *LogFact = new double [N + 1];
	LogFact[0] = 0;
	LogFact[1] = 0;
	for (int n = 2; n <= N; ++n) {
		LogFact[n] = LogFact[n - 1] + log((double) n);
	}

	// Computation
	// m ranges in [0, 2 * l]
	// m1 ranges in [0, 2 * l1]
	// m2 ranges in [0, 2 * l2]
	for (int l = 0; l <= L; ++l) {
		for (int m = 0; m < 2 * l + 1; ++m) {
			for (int l1 = 0; l1 <= L; ++l1) {
				for (int m1 = 0; m1 < 2 * l1 + 1; ++m1) {
					for (int l2 = 0; l2 <= L; ++l2) {
						for (int m2 = 0; m2 < 2 * l2 + 1; ++m2) {
							// Compute CG coefficient for tuple (l, l1, l2, m, m1, m2)
							const int index = Index(l, l1, l2, m, m1, m2, L + 1, L + 1, 2 * L + 1, 2 * L + 1, 2 * L + 1);	
							flat[index] = slowCG(l, l1, l2, m - l, m1 - l1, m2 - l2, LogFact);
						}
					}
				}
			}
		}
	}

	// Memory release
	delete[] LogFact;

	// Output
	return tensor;
}

// +---------------+
// | Sparse Tensor |
// +---------------+

std::vector<at::Tensor> sparse_tensor(const int L) {
	// Threshold for being zero
	const double threshold = 1e-6;

	// COO
	std::vector<int> l_indices;
	std::vector<int> l1_indices;
	std::vector<int> l2_indices;
	std::vector<int> m_indices;
	std::vector<int> m1_indices;
	std::vector<int> m2_indices;
	std::vector<double> values;

	l_indices.clear();
	l1_indices.clear();
	l2_indices.clear();
	m_indices.clear();
	m1_indices.clear();
	m2_indices.clear();
	values.clear();

	// Precomputatin for log of factorials
        const int N = 3 * L + 1;
        double *LogFact = new double [N + 1];
        LogFact[0] = 0;
        LogFact[1] = 0;
        for (int n = 2; n <= N; ++n) {
                LogFact[n] = LogFact[n - 1] + log((double) n);
        }
	
	// Computation
        for (int l = 0; l <= L; ++l) {
                for (int m = 0; m < 2 * l + 1; ++m) {
                        for (int l1 = 0; l1 <= L; ++l1) {
                                for (int m1 = 0; m1 < 2 * l1 + 1; ++m1) {
                                        for (int l2 = 0; l2 <= L; ++l2) {
                                                for (int m2 = 0; m2 < 2 * l2 + 1; ++m2) {
                                                        // Compute CG coefficient for tuple (l, l1, l2, m, m1, m2)
                                                        double value = slowCG(l, l1, l2, m - l, m1 - l1, m2 - l2, LogFact);
							if (std::abs(value) > threshold) {
								l_indices.push_back(l);
								l1_indices.push_back(l1);
								l2_indices.push_back(l2);
								m_indices.push_back(m);
								m1_indices.push_back(m1);
								m2_indices.push_back(m2);
								values.push_back(value);
							}
                                                }
                                        }
                                }
                        }
                }
        }

	// Number of non-zeros
	const int num_nonzeros = values.size();
	assert(l_indices.size() == num_nonzeros);
	assert(l1_indices.size() == num_nonzeros);
	assert(l2_indices.size() == num_nonzeros);
	assert(m_indices.size() == num_nonzeros);
        assert(m1_indices.size() == num_nonzeros);
        assert(m2_indices.size() == num_nonzeros);

	// Memory allocation
	at::Tensor l = torch::zeros({num_nonzeros});
	at::Tensor l1 = torch::zeros({num_nonzeros});
	at::Tensor l2 = torch::zeros({num_nonzeros});
	at::Tensor m = torch::zeros({num_nonzeros});
        at::Tensor m1 = torch::zeros({num_nonzeros});
        at::Tensor m2 = torch::zeros({num_nonzeros});
	at::Tensor v = torch::zeros({num_nonzeros});

	float *l_flat = reinterpret_cast<float*>(l.data<float>());
	float *l1_flat = reinterpret_cast<float*>(l1.data<float>());
	float *l2_flat = reinterpret_cast<float*>(l2.data<float>());
	float *m_flat = reinterpret_cast<float*>(m.data<float>());
        float *m1_flat = reinterpret_cast<float*>(m1.data<float>());
        float *m2_flat = reinterpret_cast<float*>(m2.data<float>());
	float *v_flat = reinterpret_cast<float*>(v.data<float>());

	for (int i = 0; i < num_nonzeros; i++) {
		l_flat[i] = l_indices[i];
		l1_flat[i] = l1_indices[i];
		l2_flat[i] = l2_indices[i];
		m_flat[i] = m_indices[i];
		m1_flat[i] = m1_indices[i];
		m2_flat[i] = m2_indices[i];
		v_flat[i] = values[i];
	}

	// Memory release
        delete[] LogFact;

	// Return outputs
	std::vector<at::Tensor> outputs;
	outputs.clear();
	outputs.push_back(l);
	outputs.push_back(l1);
	outputs.push_back(l2);
	outputs.push_back(m);
	outputs.push_back(m1);
	outputs.push_back(m2);
	outputs.push_back(v);
	return outputs;
}

// Test API
std::vector<at::Tensor> test_api(const std::vector<at::Tensor> &tensors) {
	const int N = tensors.size();
	std::vector<at::Tensor> result;
	for (int i = 0; i < N; ++i) {
		result.push_back(torch::zeros({}));
	}
	return result;
}

// Registration
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("dense_tensor", &dense_tensor, "Dense Tensor");
	m.def("sparse_tensor", &sparse_tensor, "Sparse Tensor");
	m.def("test_api", &test_api, "Test API");
}
