#include "transport_op.hpp"

void DiscreteToMoment::Mult(const mfem::Vector &psi, mfem::Vector &phi) const 
{
	auto psi_view = TransportVectorView(psi.GetData(), extents_psi); 
	auto phi_view = MomentVectorView(phi.GetData(), extents_phi);
	const auto num_moments = extents_phi.extent(1); 
	const auto dim = quad.Dimension(); 
	assert(num_moments==1 or num_moments == dim+1);
	phi = 0.0; 
	for (auto g=0; g<extents_psi.extent(0); g++) {
		for (auto a=0; a<extents_psi.extent(1); a++) {
			for (auto s=0; s<extents_psi.extent(2); s++) {
				phi_view(g,0,s) += quad.GetWeight(a) * psi_view(g,a,s); 
			}
		}
	}

	if (num_moments>1) {
		for (auto g=0; g<extents_psi.extent(0); g++) {
			for (auto a=0; a<extents_psi.extent(1); a++) {
				const auto &Omega = quad.GetOmega(a); 
				for (auto s=0; s<extents_psi.extent(2); s++) {
					for (auto m=0; m<dim; m++) {
						phi_view(g,1+m,s) += Omega(m) * quad.GetWeight(a) * psi_view(g,a,s); 						
					}
				}
			}
		}		
	}
}

void DiscreteToMoment::MultTranspose(const mfem::Vector &phi, mfem::Vector &psi) const 
{
	auto psi_view = TransportVectorView(psi.GetData(), extents_psi); 
	auto phi_view = MomentVectorView(phi.GetData(), extents_phi);
	const auto dim = quad.Dimension(); 
	const auto num_moments = extents_phi.extent(1); 
	assert(num_moments==1 or (dim==2 and num_moments == 3) or (dim==3 and num_moments==4)); 
	for (auto g=0; g<extents_psi.extent(0); g++) {
		for (auto a=0; a<extents_psi.extent(1); a++) {
			for (auto s=0; s<extents_psi.extent(2); s++) {
				psi_view(g,a,s) = phi_view(g,0,s) / quad.SumWeights(); 
			}
		}
	}

	if (num_moments>1) {
		for (auto g=0; g<extents_psi.extent(0); g++) {
			for (auto a=0; a<extents_psi.extent(1); a++) {
				const auto &Omega = quad.GetOmega(a); 
				for (auto s=0; s<extents_psi.extent(2); s++) {
					for (auto m=0; m<dim; m++) {
						psi_view(g,a,s) += 3 * Omega(m) * phi_view(g,1+m,s) / quad.SumWeights(); 						
					}
				}
			}
		}
	}
}

void DiscreteToMoment::AddMultTranspose(const mfem::Vector &phi, mfem::Vector &psi, double alpha) const 
{
	auto psi_view = TransportVectorView(psi.GetData(), extents_psi); 
	auto phi_view = MomentVectorView(phi.GetData(), extents_phi);
	const auto dim = quad.Dimension(); 
	const auto num_moments = extents_phi.extent(1); 
	assert(num_moments==1 or (dim==2 and num_moments == 3) or (dim==3 and num_moments==4)); 
	for (auto g=0; g<extents_psi.extent(0); g++) {
		for (auto a=0; a<extents_psi.extent(1); a++) {
			for (auto s=0; s<extents_psi.extent(2); s++) {
				psi_view(g,a,s) += alpha*phi_view(g,0,s) / quad.SumWeights(); 
			}
		}
	}

	if (num_moments>1) {
		for (auto g=0; g<extents_psi.extent(0); g++) {
			for (auto a=0; a<extents_psi.extent(1); a++) {
				const auto &Omega = quad.GetOmega(a); 
				for (auto s=0; s<extents_psi.extent(2); s++) {
					for (auto m=0; m<dim; m++) {
						psi_view(g,a,s) += alpha * 3 * Omega(m) * phi_view(g,1+m,s) / quad.SumWeights(); 						
					}
				}
			}
		}
	}
}