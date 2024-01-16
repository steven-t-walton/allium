#include "transport_op.hpp"

void DiscreteToMoment::Mult(const mfem::Vector &psi, mfem::Vector &phi) const 
{
	auto psi_view = TransportVectorView(psi.GetData(), extents_psi); 
	auto phi_view = MomentVectorView(phi.GetData(), extents_phi);
	phi = 0.0; 
	for (auto g=0; g<extents_psi.extent(0); g++) {
		for (auto a=0; a<extents_psi.extent(1); a++) {
			for (auto s=0; s<extents_psi.extent(2); s++) {
				phi_view(0,g,s) += quad.GetWeight(a) * psi_view(g,a,s); 
			}
		}
	}
}

void DiscreteToMoment::MultTranspose(const mfem::Vector &phi, mfem::Vector &psi) const 
{
	auto psi_view = TransportVectorView(psi.GetData(), extents_psi); 
	auto phi_view = MomentVectorView(phi.GetData(), extents_phi);
	for (auto g=0; g<extents_psi.extent(0); g++) {
		for (auto a=0; a<extents_psi.extent(1); a++) {
			for (auto s=0; s<extents_psi.extent(2); s++) {
				psi_view(g,a,s) = phi_view(0,g,s) / quad.SumWeights(); 
			}
		}
	}
}