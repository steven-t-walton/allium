#include "config.hpp"
#include "angular_quadrature.hpp"
#include "tvector.hpp"
#include <regex>
#include "yaml-cpp/yaml.h"

int AngularQuadrature::GetIndexForAngle(const mfem::Vector &angle) const 
{
	auto vec_eq = [&angle](const mfem::Vector &x) {
		double diff = 0.0; 
		for (auto d=0; d<angle.Size(); d++) {
			diff += pow(x(d) - angle(d), 2); 
		}
		return sqrt(diff) < 1e-14; 
	};
	auto it = std::find_if(Omegas.begin(), Omegas.end(), vec_eq); 
	assert(it != Omegas.end()); 
#ifndef NDEBUG 
	auto Omega = GetOmega(std::distance(Omegas.begin(), it)); 
	Omega -= angle; 
	assert(Omega.Norml2() < 1e-13); 
#endif
	return std::distance(Omegas.begin(), it); 
}

int AngularQuadrature::GetReflectedAngleIndex(int angle, const mfem::Vector &nor) const 
{
	const auto &Omega = GetOmega(angle); 
	mfem::Vector Omegap(Omega); 
	double dot2 = 2*(Omega*nor); 
	for (auto d=0; d<dim; d++) {
		Omegap(d) -= dot2 * nor(d); 
	}
	return GetIndexForAngle(Omegap); 
}

LevelSymmetricQuadrature::LevelSymmetricQuadrature(int _order, int _dim) 
	: AngularQuadrature(_dim) 
{
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

AbuShumaysQuadrature::AbuShumaysQuadrature(int _order, int _dim) 
	: AngularQuadrature(_dim) 
{
	if (dim != 2) MFEM_ABORT("Abu Shumays only works in 2D"); 
	auto file_name = std::string(AS_QUADRATURE_DIR) + "/Q" + std::to_string(_order) + ".yaml"; 
	// check if it exists 
	std::ifstream inp(file_name); 
	if (inp.fail()) { 
		MFEM_ABORT("Abu-Shumays quadrature order " << _order << " not found at " << AS_QUADRATURE_DIR); 
	}
	inp.close(); 

	// load with yaml 
	auto node = YAML::LoadFile(file_name); 
	auto mu = node["mu"].as<std::vector<double>>();
	auto eta = node["eta"].as<std::vector<double>>(); 
	auto w = node["w"].as<std::vector<double>>();  
	const auto oct_size = mu.size(); 
	const auto num_dirs = 4 * oct_size; 
	assert(eta.size() == oct_size and w.size() == oct_size); 
	Omegas.resize(num_dirs, mfem::Vector(dim)); 
	weights.resize(num_dirs, 0.0); 

	std::array<double,2> oct_dir = {-1.0, 1.0}; 
	int angle_count = 0; 
	for (int i=0; i<oct_dir.size(); i++) {
		for (int j=0; j<oct_dir.size(); j++) {
			for (int idx=0; idx<oct_size; idx++) {
				auto &Omega = Omegas[angle_count]; 
				Omega(0) = oct_dir[i] * mu[idx]; 
				Omega(1) = oct_dir[j] * eta[idx]; 
				weights[angle_count] = w[idx]; 
				angle_count++; 
			}
		}
	}

	weights_sum = std::accumulate(weights.begin(), weights.end(), 0.0); 
	const auto conversion = 4.0*M_PI/weights_sum; 
	for (auto &w : weights) {
		w *= conversion; 
	}
	weights_sum = std::accumulate(weights.begin(), weights.end(), 0.0); 
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