#include "tvector.hpp"

void ProjectPsi(const mfem::FiniteElementSpace &fes, const AngularQuadrature &quad, 
	PhaseSpaceCoefficient &f, TransportVectorView psi)
{
	mfem::Array<int> dofs; 
	mfem::Vector vals; 
	for (auto a=0; a<quad.Size(); a++) {
		const auto &Omega = quad.GetOmega(a); 
		f.SetAngle(Omega); 
		for (auto e=0; e<fes.GetNE(); e++) {
			fes.GetElementDofs(e, dofs); 
			vals.SetSize(dofs.Size()); 
			fes.GetFE(e)->Project(f, *fes.GetElementTransformation(e), vals); 
			for (auto i=0; i<vals.Size(); i++) {
				psi(0,a,dofs[i]) = vals[i]; 
			}
		}
	}	
}

void ProjectPsi(const mfem::FiniteElementSpace &fes, const AngularQuadrature &quad, 
	const mfem::Array<double> &energy_grid, PhaseSpaceCoefficient &f, mfem::Vector &psi) 
{
	TransportVectorExtents psi_ext(energy_grid.Size()-1, quad.Size(), fes.GetVSize()); 
	assert(psi.Size() == TotalExtent(psi_ext)); 
	auto psi_view = TransportVectorView(psi.GetData(), psi_ext); 
	mfem::Array<int> dofs; 
	mfem::Vector vals; 
	const auto G = energy_grid.Size() - 1; 
	for (int g=0; g<G; g++) {
		double energy = (energy_grid[g+1] + energy_grid[g])/2; 
		f.SetEnergy(energy); 
		for (int a=0; a<quad.Size(); a++) {
			const auto &Omega = quad.GetOmega(a); 
			f.SetAngle(Omega); 
			for (int e=0; e<fes.GetNE(); e++) {
				fes.GetElementDofs(e, dofs); 
				vals.SetSize(dofs.Size()); 
				fes.GetFE(e)->Project(f, *fes.GetElementTransformation(e), vals); 
				for (auto n=0; n<vals.Size(); n++) {
					psi_view(g,a,dofs[n]) = vals[n]; 
				}
			}
		}
	}
}