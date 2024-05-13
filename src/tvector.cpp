#include "tvector.hpp"

SNTimeMassMatrix::SNTimeMassMatrix(const mfem::FiniteElementSpace &fes, 
	const TransportVectorExtents &ext, bool lump)
	: fes(fes), psi_ext(ext) 
{
	mats.SetSize(fes.GetNE()); 
	mfem::Array<int> dofs; 
	mfem::MassIntegrator mi; 
	for (int e=0; e<fes.GetNE(); e++) {
		fes.GetElementDofs(e, dofs); 
		mats[e] = new mfem::DenseMatrix(dofs.Size()); 
		mi.AssembleElementMatrix(*fes.GetFE(e), *fes.GetElementTransformation(e), *mats[e]); 
		if (lump)
			mats[e]->Lump(); 
	}
}

void SNTimeMassMatrix::Mult(const mfem::Vector &psi, mfem::Vector &Mpsi) const 
{
	auto psi_view = ConstTransportVectorView(psi.GetData(), psi_ext); 
	auto Mpsi_view = TransportVectorView(Mpsi.GetData(), psi_ext); 
	mfem::Array<int> dofs; 
	mfem::Vector psi_local, Mpsi_local; 
	for (int g=0; g<psi_ext.extent(0); g++) {
		for (int a=0; a<psi_ext.extent(1); a++) {
			for (int e=0; e<fes.GetNE(); e++) {
				fes.GetElementDofs(e, dofs); 
				const auto &elmat = *mats[e]; 
				psi_local.SetSize(dofs.Size()); 
				Mpsi_local.SetSize(dofs.Size()); 					
				for (int n=0; n<dofs.Size(); n++) psi_local(n) = psi_view(g,a,dofs[n]); 
				elmat.Mult(psi_local, Mpsi_local); 
				for (int n=0; n<dofs.Size(); n++) Mpsi_view(g,a,dofs[n]) = Mpsi_local(n); 
			}
		}
	}
}

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