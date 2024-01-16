#include "config.hpp"
#include "angular_quadrature.hpp"
#include <regex>

LevelSymmetricQuadrature::LevelSymmetricQuadrature(int _order, int _dim) : dim(_dim) {
	std::string file_name = std::string(LS_QUADRATURE_DIR) + "/LS_" + std::to_string(_order) + ".txt"; 
	std::ifstream inp(file_name); 
	if (inp.fail()) { MFEM_ABORT("quadrature order " << _order << " not found at " << LS_QUADRATURE_DIR); }

	std::string line; 
	std::getline(inp, line); 
	std::regex num_dirs_regex("Total Directions = ([0-9]+)"); 
	std::smatch num_dirs_match; 
	std::regex_search(line, num_dirs_match, num_dirs_regex); 
	int num_dirs = std::stoi(num_dirs_match[1]); 
	if (dim<3) num_dirs /= 2; 
	Omegas.resize(num_dirs, mfem::Vector(dim)); 
	weights.resize(num_dirs, 0.0); 
	double omega[3]; 
	double w; 
	for (int i=0; i<num_dirs; i++) {
		std::getline(inp, line); 
		std::istringstream ss(line); 
		ss >> omega[0] >> omega[1] >> omega[2] >> w; 
		for (int d=0; d<dim; d++) {
			Omegas[i](d) = omega[d]; 
		}
		if (dim<3) w *= 2;  
		weights[i] = w;
	}
	for (auto w : weights) {
		weights_sum += w; 
	}
#ifndef NDEBUG 
	if (abs(weights_sum - 4*M_PI) > 1e-7) {
		MFEM_ABORT("quadrature file " << file_name << " has weights sum to " << weights_sum); 
	} 
#endif 
}

GaussLegendreQuadratureRule::GaussLegendreQuadratureRule(int order, int dim)
{
	if (dim != 1) { MFEM_ABORT("Gauss Legendre only setup for 1D"); }
	const auto &rule = mfem::IntRules.Get(mfem::Geometry::SEGMENT, order); 
	const auto size = rule.GetNPoints(); 
	Omegas.resize(size, mfem::Vector(dim)); 
	weights.resize(size); 
	for (auto n=0; n<size; n++) {
		const auto &ip = rule.IntPoint(n); 
		Omegas[n](0) = 2*ip.x - 1; 
		weights[n] = 2*ip.weight; 
	}
	for (const auto &w : weights) {
		weights_sum += w; 
	}
}

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

double ComputeAlpha(const AngularQuadrature &quad, const mfem::Vector &dir) 
{
	mfem::Vector nor(dir.Size()); 
	nor.Set(1./dir.Norml2(), dir); 
	double alpha = 0.0; 
	for (int angle=0; angle<quad.Size(); angle++) {
		const auto &Omega = quad.GetOmega(angle); 
		alpha += std::fabs(Omega * nor) * quad.GetWeight(angle); 
	}
	return alpha/quad.SumWeights(); 
}