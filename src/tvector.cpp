#include "tvector.hpp"
#include "multigroup.hpp"
#include "lumping.hpp"

ZerothMomentCoefficient::ZerothMomentCoefficient(
	const mfem::FiniteElementSpace &fes, const MomentVectorExtents &phi_ext, const mfem::Vector &data)
	: fes(fes), phi_ext(phi_ext), data(data), mfem::VectorCoefficient(phi_ext.extent(MomentIndex::ENERGY))
{
}

void ZerothMomentCoefficient::Eval(
	mfem::Vector &v, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip)
{
	auto view = MomentVectorView(data.GetData(), phi_ext);
	v.SetSize(vdim);
	mfem::Array<int> dofs;
	const auto *fe = fes.GetFE(trans.ElementNo);
	const auto dof = fe->GetDof();
	shape.SetSize(dof);
	local_data.SetSize(dof);
	fe->CalcShape(ip, shape);
	fes.GetElementDofs(trans.ElementNo, dofs);
	for (int g=0; g<vdim; g++) {
		for (int i=0; i<dof; i++) {
			local_data(i) = view(g,moment_id,dofs[i]);
		}
		v(g) = shape * local_data;
	}
}

FirstMomentCoefficient::FirstMomentCoefficient(
	const mfem::FiniteElementSpace &fes, const MomentVectorExtents &ext, const mfem::Vector &data)
	: fes(fes), ext(ext), data(data), 
	  mfem::MatrixCoefficient(ext.extent(MomentIndex::ENERGY), fes.GetMesh()->Dimension())
{
	if (ext.extent(MomentIndex::MOMENT) < width+1)
		MFEM_ABORT("not enough data");
}

void FirstMomentCoefficient::Eval(mfem::DenseMatrix &K, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip)
{
	auto view = ConstMomentVectorView(data.GetData(), ext);
	K.SetSize(height, width);
	K = 0.0;
	mfem::Array<int> dofs; 
	const auto &fe = *fes.GetFE(trans.ElementNo);
	const auto dof = fe.GetDof();
	shape.SetSize(dof);
	fe.CalcShape(ip, shape);
	fes.GetElementDofs(trans.ElementNo, dofs);
	for (int g=0; g<height; g++) {
		for (int d=0; d<width; d++) {
			for (int i=0; i<dof; i++) {
				K(g,d) += view(g,d+1,dofs[i]) * shape(i);
			}
		}
	}
}

SNTimeMassMatrix::SNTimeMassMatrix(const mfem::FiniteElementSpace &fes, 
	const TransportVectorExtents &ext, bool lump)
	: fes(fes), psi_ext(ext) 
{
	mats.SetSize(fes.GetNE()); 
	mfem::Array<int> dofs; 
	mfem::MassIntegrator mi; 
	for (int e=0; e<fes.GetNE(); e++) {
		auto &trans = *fes.GetElementTransformation(e);
		LumpedIntegrationRule lumped_ir(trans.GetGeometryType()); 
		if (lump) mi.SetIntegrationRule(lumped_ir);
		mats[e] = new mfem::DenseMatrix;
		mi.AssembleElementMatrix(*fes.GetFE(e), trans, *mats[e]); 
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
		f.SetEnergy(energy_grid[g], energy_grid[g+1], (energy_grid[g] + energy_grid[g+1])/2); 
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

void ProjectPsi(const mfem::FiniteElementSpace &fes, const AngularQuadrature &quad, 
	const MultiGroupEnergyGrid &energy_grid, PhaseSpaceCoefficient &f, mfem::Vector &psi)
{
	TransportVectorExtents psi_ext(energy_grid.Size(), quad.Size(), fes.GetVSize());
	assert(psi.Size() == TotalExtent(psi_ext));
	auto psi_view = TransportVectorView(psi.GetData(), psi_ext);
	mfem::Array<int> dofs; 
	mfem::Vector vals; 
	const auto G = energy_grid.Size();
	for (int g=0; g<G; g++) {
		f.SetEnergy(energy_grid.LowerBound(g), energy_grid.UpperBound(g), energy_grid.MeanEnergy(g));
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