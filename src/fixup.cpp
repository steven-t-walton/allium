#include "fixup.hpp"
#include "log.hpp"

bool ZeroFixupOperator::Apply(const mfem::DenseMatrix &A, const mfem::Vector &rhs, mfem::Vector &solution) const
{
	bool applied = false;
	for (auto &x : solution) {
		if (x <= minimum_solution) {
			x = minimum_solution; 
			applied = true;
		}
	}
	if (applied) EventLog.Register("fixup applied");
	return applied;
}

bool ZeroAndScaleFixupOperator::Apply(const mfem::DenseMatrix &A, 
	const mfem::Vector &rhs, mfem::Vector &solution) const 
{
	const auto norm = solution.Norml2();
	const auto rel_min = minimum_solution * norm;
	if (solution.Min() > minimum_solution) {
		return false; 
	}

	const int height = A.Height();
	weights.SetSize(height); 
	for (int i=0; i<height; i++) {
		weights(i) = 0.0;
		for (int j=0; j<height; j++) {
			weights(i) += A(j,i);			
		}
	}

	int nff_count = 0;
	for (int i=0; i<solution.Size(); i++) {
		if (solution(i) <= minimum_solution) {
			solution(i) = minimum_solution;
			nff_count++;
		}
	}

	const double original_balance = rhs.Sum();
	const double new_balance = weights * solution;
	if (original_balance < 0.0 or new_balance < 0.0) {
		EventLog.Register("fixup failed");
	}

	else if (original_balance == 0.0 or new_balance == 0.0) {
		solution = 0.0;
	}

	else {
		solution *= (original_balance/new_balance); 
		EventLog.Register("fixup applied");
	}

	return true;
}

bool LocalOptimizationFixupOperator::Apply(const mfem::DenseMatrix &A, 
	const mfem::Vector &rhs, mfem::Vector &solution) const 
{
	const double original_balance = rhs.Sum();
	if (solution.Min() > minimum_solution or original_balance <= minimum_solution) {
		return false; 
	}

	const int height = A.Height();
	ones.SetSize(height); 
	weights.SetSize(height); 
	ones = 1.0; 
	A.MultTranspose(ones, weights); 

	low.SetSize(height); 
	high.SetSize(height); 
	low = minimum_solution; 
	high = std::numeric_limits<double>::max(); 

	opt.SetBounds(low, high); 
	opt.SetLinearConstraint(weights, original_balance); 
	source = solution;
	opt.Mult(source, solution); 
	if (opt.GetConverged()) {
		EventLog.Register("fixup applied");
	} else {
		EventLog.Register("fixup failed");
	}
	return true;
}

bool RyosukeFixupOperator::Apply(const mfem::DenseMatrix &A, const mfem::Vector &rhs, mfem::Vector &solution) const
{
	const auto sz = solution.Size(); 
	const auto count = perform_nff(sz, solution.GetData(), A.GetData(), rhs.GetData());
	return count > 0;
}

unsigned RyosukeFixupOperator::perform_nff(unsigned sz, double *x, double *A, double *b) const
{
	unsigned nff_cnt;
	std::vector<unsigned> count(sz);
	unsigned pos(0);

	for (unsigned i = 0; i < sz; ++i) {
		if (x[i] <= minimum_solution)
			x[i] = minimum_solution;
		else {
			count[pos++] = i; //.push_back(i);
		}
	} // i

	//        unsigned pos = count.size();

	if (pos == 0) {
		return 0;
	}

	else if (pos == sz) {
		return 0;
	}

	// form the new matrix equation, Cx=d
	std::vector<double> C(pos * pos, 0.);
	std::vector<double> d(pos, 0.);
	for (unsigned i = 0; i < pos; ++i) {
		unsigned ii = count[i];

		C[i] = A[ii*sz];
		for (unsigned j = 1; j < sz; j++) {
			C[i] += A[ii*sz + j]; // the first row
		}
		if (i == 0) {
			for (unsigned j = 1; j < pos; j++) {
				C[j * pos] = -x[count[j]] / x[count[0]]; // the first column
			}
		} else
			C[i * (pos + 1)] = 1.; // the diagonal
	}
	for (unsigned j = 0; j < sz; j++) {
		d[0] += b[j]; // the total source
	}

	if (pos == 1) {
		x[count[0]] = d[0] / C[0];
	} else if (pos == 2) {
		double det = C[0] * C[3] - C[1] * C[2];
		det = 1. / det;
		x[count[0]] = d[0] * C[3] * det;
		x[count[1]] = -d[0] * C[2] * det;
	} else if (pos == 3) {
		double det = C[0] * C[4] * C[8] - C[1] * C[3] * C[8] - C[2] * C[4] * C[6];
		det = 1. / det;
		x[count[0]] = d[0] * C[4] * C[8] * det;
		x[count[1]] = -d[0] * C[3] * C[8] * det;
		x[count[2]] = -d[0] * C[4] * C[6] * det;
	} else {
		MFEM_ABORT("pos > 3 not supported");
	}

	// if there are still negative fluxes, zero all fluxes and return
	bool rebalance_fail = false;
	for (unsigned i = 0; i < pos; ++i) {
		if (x[count[i]] < minimum_solution) {
			rebalance_fail = true;
			for (unsigned j = 0; j < pos; ++j)
				x[count[j]] = minimum_solution;
			break;
		}
	}
	if (rebalance_fail) {
		EventLog.Register("rebalance failed");
	} else {
		EventLog.Register("fixup applied");
	}

	return 1;
}