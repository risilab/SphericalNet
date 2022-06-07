#include "ClebschGordanCE.h"

using namespace std;

int main(int argc, char **argv) {
	// Initialization of the CG table
	const int L = 5;
	ClebschGordanCE *obj = new ClebschGordanCE(5);

	// Fast query given (l, l1, l2, m, m1, m2)
	const int l = 2;
	const int l1 = 1;
	const int l2 = 1;
	const int m = -1; // Ranging from -l to l
	const int m1 = 1; // Ranging from -l1 to l1
	const int m2 = 0; // Ranging from -l2 to l2
	const double value = obj -> get(l, l1, l2, m, m1, m2);
	printf("%.6f\n", value);
	return 0;
}
