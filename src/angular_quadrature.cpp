#include "config.hpp"
#include "angular_quadrature.hpp"
#include "tvector.hpp"
#include <regex>

LevelSymmetricQuadrature::LevelSymmetricQuadrature(int _order, int _dim) : AngularQuadrature(_dim) {
	if (dim==1) {
		int degree = 2*_order - 1; 
		const auto &rule = mfem::IntRules.Get(mfem::Geometry::SEGMENT, degree); 
		const auto size = rule.GetNPoints(); 
		Omegas.resize(size, mfem::Vector(dim)); 
		weights.resize(size); 
		for (auto n=0; n<size; n++) {
			const auto &ip = rule.IntPoint(n); 
			Omegas[size-n-1](0) = 2*ip.x - 1; // positive angles first to match structure of files 
			weights[n] = 4*M_PI*ip.weight; 
		}
	}

	else {
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
	}
	
	for (auto w : weights) {
		weights_sum += w; 
	}
#ifndef NDEBUG 
	if (abs(weights_sum - 4*M_PI) > 1e-7) {
		MFEM_ABORT("quadrature order " << _order << ", dimension " << dim << " has weights that don't sum to 4pi"); 
	} 
#endif 
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