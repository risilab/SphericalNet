// Framework: GraphFlow
// Class: ClebschGordanCE
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __CLEBSCHGORDANCE_H_INCLUDED__
#define __CLEBSCHGORDANCE_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <complex>
#include <assert.h>

using namespace std;

class ClebschGordanCE {
public:

	ClebschGordanCE(int L) {
		this -> L = L;

		N = 3 * L + 1;
		LogFact = new double [N + 1];
		LogFact[0] = 0;
		LogFact[1] = 0;
		for (int n = 2; n <= N; ++n) {
			LogFact[n] = LogFact[n - 1] + log((double) n);
		}

		// Bigtable

		cout << "[Allocate memory for CG coefficients]" << endl;

		bigtable = new double***** [L + 1];
		for (int l = 0; l <= L; ++l) {
			bigtable[l] = new double**** [2 * l + 1];
			for (int m = 0; m < 2 * l + 1; ++m) {
				bigtable[l][m] = new double*** [L + 1];
				for (int l1 = 0; l1 <= L; ++l1) {
					bigtable[l][m][l1] = new double** [2 * l1 + 1];
					for (int m1 = 0; m1 < 2 * l1 + 1; ++m1) {
						bigtable[l][m][l1][m1] = new double* [L + 1];
						for (int l2 = 0; l2 <= L; ++l2) {
							bigtable[l][m][l1][m1][l2] = new double [2 * l2 + 1];
						}
					}
				}
			}
		}

		cout << "[Done allocating memory for CG coefficients]" << endl;

		cout << "[Precompute all necessary CG coefficients]" << endl;

		for (int l = 0; l <= L; ++l) {
			for (int m = 0; m < 2 * l + 1; ++m) {
				for (int l1 = 0; l1 <= L; ++l1) {
					for (int m1 = 0; m1 < 2 * l1 + 1; ++m1) {
						for (int l2 = 0; l2 <= L; ++l2) {
							for (int m2 = 0; m2 < 2 * l2 + 1; ++m2) {
								bigtable[l][m][l1][m1][l2][m2] = slowCG(l1, l2, l, m1 - l1, m2 - l2, m - l);
							}
						}
					}
				}
			}
		}

		cout << "[Done precomputation for CG coefficients]" << endl;
	}

	// DEPRECATED
	/*
	ClebschGordanCE(int L) {
		this -> L = L;

		// Initialization
		group = 0;
		allocate();
		computeRecursively();
	}
	*/

	// DEPRECATED
	/*
	void allocate(){
		double***** Tl1 = new double**** [L + 1];
		table = Tl1;
		for(int l1 = 0; l1 <= L; ++l1) {
			double**** Tl2 = new double*** [l1 + 1];
			Tl1[l1] = Tl2;
			for(int l2 = 0; l2 <= l1; ++l2) {
				int loffset = l1 - l2;
				double*** Tl = new double** [min(l1 + l2, L) - loffset + 1];
				Tl2[l2] = Tl;
				for(int l = loffset; l <= min(l1 + l2, L); ++l) {
					double** Tm1 = new double* [2 * l1 + 1];
					Tl[l - loffset] = Tm1;
					for(int m1 = -l1; m1 <= l1; ++m1) {
						int m2offset = max(-l2, -l - m1);
						double *Tm2 = new double [min(l2, l - m1) - m2offset + 1];
						Tm1[m1 + l1] = Tm2;
					}
				}
			}
		}
	}
	*/

	// DEPRECATED
	/*
	int computeRecursively(){
		for(int l1 = 0; l1 <= L; ++l1) {
			double**** Tl2 = table[l1];
			for(int l2 = 0; l2 <= l1; ++l2) {
				double*** Tl = Tl2[l2];
				int loffset = l1 - l2;
				for(int l = loffset; l <= min(l1 + l2, L); ++l) {
					double** Tm1 = Tl[l - loffset];
					double* c; c = new double [2 * l1 + 2];
					double* nc; nc = new double [2 * l1 + 2];
	
					if (1) {
						int m = l;
						for(int m1 = -l1; m1 <= l1; ++m1) {
							int m2 = m - m1;
							if ((m2 >= -l2) && (m2 <= l2)) {
								c[m1 + l1] = slowCG(l1, l2, l, m1, m2, m);
								Tm1[m1 + l1][m2 - max(-l2, -l - m1)] = c[m1 + l1];
							} else {
								c[m1 + l1] = 0;
							}
						}
						c[2 * l1 + 1] = 0; 
						nc[2 * l1 + 1] = 0;  
					}

					for (int m = l - 1; m >= -l; --m) {
						for (int m1 = -l1; m1 <= l1; ++m1) {
							int m2 = m - m1;
							if ((m2 >= -l2) && (m2 <= l2)) {
								nc[m1 + l1] = (c[m1 + l1 + 1] * sqrt((double)(l1 + m1 + 1) * (l1 - m1)) + c[m1 + l1] * sqrt((double)(l2 + m2 + 1) * (l2 - m2))) / (sqrt((double)(l + m + 1) * (l - m)));
								Tm1[m1 + l1][m2 - max(-l2, -l - m1)] = nc[m1 + l1];
							} else {
								nc[m1 + l1] = 0;
							}
						}
						double* t = c; c = nc; nc = t;
					}

					// delete[] c;
					// delete[] nc;
				}
			}
		}
	}
	*/

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

	double deprecated_slowCG(int l1, int l2, int l, int m1, int m2, int m) {
		int m3 = -m;
		int t1 = l2 - m1 - l;
		int t2 = l1 + m2 - l;
		int t3 = l1 + l2 - l;
		int t4 = l1 - m1;
		int t5 = l2 + m2;

		int tmin = max(0, max(t1, t2));
		int tmax = min(t3, min(t4, t5));

		double wigner = 0;

		double logA = (logfact(l1 + l2 - l) + logfact(l1 - l2 + l) + logfact(-l1 + l2 + l) - logfact(l1 + l2 + l + 1)) / 2;
		logA += (logfact(l1 + m1) + logfact(l1 - m1) + logfact(l2 + m2) + logfact(l2 - m2) + logfact(l + m3) + logfact(l - m3)) / 2;

		for (int t = tmin; t <= tmax; ++t) {
			double logB = logfact(t) + logfact(t - t1) + logfact(t - t2) + logfact(t3 - t) + logfact(t4 - t) + logfact(t5 - t);
			wigner += plusminus(t) * exp(logA - logB);
		}

		return plusminus(l1 - l2 - m3) * plusminus(l1 - l2 + m) * sqrt((double)(2 * l + 1)) * wigner; 
	}

	double slowCG(int l1, int l2, int l, int m1, int m2, int m) {
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

	double get(int l, int l1, int l2, int m, int m1, int m2) {
                return bigtable[l][m + l][l1][m1 + l1][l2][m2 + l2];
        }

	/*
	double get(int l, int m, int l1, int m1, int l2, int m2) {
		return bigtable[l][m + l][l1][m1 + l1][l2][m2 + l2];
	}
	*/

	// DEPRECATED
	/*
	double get(int l, int m, int l1, int m1, int l2, int m2) {
		double inverter = 1;

		if ((l1 < 0) || (l2 < 0) || (l < 0)) {
			return 0;
		}

		if ((l1 > L) || (l2 > L) || (l > L)) {
			return 0;
		}
  
		if ((m1 < -l1) || (m1 > l1)) {
			return 0;
		}

		if ((m2 < -l2) || (m2 > l2)) {
			return 0;
		}

  		if ((m < -l) || (m > l)) {
  			return 0;
  		}

		if (l2 > l1) {
			int t;
			t = l1; l1 = l2; l2 = t;
			t = m1; m1 = m2; m2 = t;
			if ((l1 + l2 - l) % 2 == 1) {
				inverter = -1;
			}
		}

		if (l < l1 - l2) {
			return 0;
		}
  
		if (m != m1 + m2) {
			return 0;
		}
  
		if (m2 < 0) {
			m1 = -m1;
			m2 = -m2;
			if ((l1 + l2 - l) % 2 == 1) {
				inverter *= -1;
			}
		}

		int loffset = l1 - l2;
		int m2offset = max(-l2, -l - m1);

		return table[l1][l2][l - loffset][m1 + l1][m2 - m2offset] * inverter;
	}
	*/

	// DEPRECATED
	/*
	int group;
	double***** table;
	*/

	int L;
	int N;
	double *LogFact;
	double ******bigtable;

	~ClebschGordanCE() {
		for (int l = 0; l <= L; ++l) {
			for (int m = 0; m < 2 * l + 1; ++m) {
				for (int l1 = 0; l1 <= L; ++l1) {
					for (int m1 = 0; m1 < 2 * l1 + 1; ++m1) {
						for (int l2 = 0; l2 <= L; ++l2) {
							delete[] bigtable[l][m][l1][m1][l2];
						}
						delete[] bigtable[l][m][l1][m1];
					}
					delete[] bigtable[l][m][l1];
				}
				delete[] bigtable[l][m];
			}
			delete[] bigtable[l];
		}
		delete[] bigtable;

		// DEPRECATED
		/*
		for (int l1 = 0; l1 <= L; ++l1) {
			for (int l2 = 0; l2 <= l1; ++l2) {
				int loffset = l1 - l2;
				for (int l = loffset; l <= min(l1 + l2, L); ++l) {
					for (int m1 = -l1; m1 <= l1; ++m1){
						delete[] table[l1][l2][l - loffset][m1 + l1];
					}
					delete[] table[l1][l2][l - loffset]; 
				}
				delete[] table[l1][l2];
			}
			delete[] table[l1];
		}
		delete[] table;
		*/
	}
};

#endif
