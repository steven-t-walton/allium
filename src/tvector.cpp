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