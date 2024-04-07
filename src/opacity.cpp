#include "opacity.hpp"

void GroupCollapseOperator::Mult(const mfem::Vector &sigma_mf, mfem::Vector &sigma_gray) const 
{
	const int G = fes.GetVDim(); 
	const auto ordering = fes.GetOrdering(); 
	assert(ordering == mfem::Ordering::Type::byNODES); 
	assert(sigma_gray.Size() == height); 
	sigma_gray = 0.0; 
	mfem::Vector F(G); 
	F = 1.0; 
	mfem::Array<int> dof; 
	for (int e=0; e<fes.GetNE(); e++) {
		fes.GetElementDofs(e, dof); 
		auto &trans = *fes.GetMesh()->GetElementTransformation(e); 
		const auto &fe = *fes.GetFE(e); 
		const auto &ir = fe.GetNodes(); 
		for (int n=0; n<ir.Size(); n++) {
			if (f) 
				f->Eval(F, trans, ir.IntPoint(n)); 
			double denom = 0.0; 
			for (int g=0; g<G; g++) {
				double dE = energy_grid[g+1] - energy_grid[g]; 
				sigma_gray(dof[n]) += F[g] * sigma_mf(dof[n] + g*height) * dE; 
				denom += F[g] * dE; 
			}
			sigma_gray(dof[n]) /= denom; 
		}
	}
}