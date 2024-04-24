#include "fixup.hpp"
#include "log.hpp"

void ZeroAndScaleFixupOperator::Mult(const mfem::Vector &solution, mfem::Vector &fixed) const 
{
	if (!DoFixup(solution)) {
		fixed = solution; 
		return; 
	}

	ones.SetSize(height); 
	weights.SetSize(height); 
	ones = 1.0; 
	double original_balance = (*rhs) * ones; 
	A->MultTranspose(ones, weights); 

	for (int i=0; i<solution.Size(); i++) {
		fixed(i) = std::max(minimum_solution, solution(i)); 
	}

	if (original_balance < 0.0) {
		EventLog["fixup failed"] += 1; 
	} else {
		double new_balance = weights * fixed; 
		fixed *= (original_balance/new_balance); 		
		EventLog["fixup applied"] += 1; 
	}
}

void LocalOptimizationFixupOperator::Mult(const mfem::Vector &solution, mfem::Vector &fixed) const 
{
	if (!DoFixup(solution)) {
		fixed = solution; 
		return; 
	}

	ones.SetSize(height); 
	weights.SetSize(height); 
	ones = 1.0; 
	A->MultTranspose(ones, weights); 
	double original_balance = (*rhs) * ones; 

	low.SetSize(height); 
	high.SetSize(height); 
	low = minimum_solution; 
	high = std::numeric_limits<double>::max(); 

	opt.SetBounds(low, high); 
	opt.SetLinearConstraint(weights, original_balance); 
	opt.Mult(solution, fixed); 
	if (opt.GetConverged()) {
		EventLog["fixup applied"] += 1; 
	} else {
		EventLog["fixup failed"] += 1; 
	}
}