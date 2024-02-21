#include "config.hpp"
#include "angular_quadrature.hpp"
#include "tvector.hpp"
#include <regex>

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

void GetTangentVectors(const mfem::Vector &nor, mfem::DenseMatrix &tau) {
	const auto dim = nor.Size(); 
	if (dim != 2 and dim != 3) { MFEM_ABORT("only defined for dim=2 or dim=3"); }
	tau.SetSize(dim, dim-1); 
	if (dim==2) {
		tau(0,0) = nor(1); 
		tau(1,0) = -nor(0); 
	}

	else if (dim==3) {
		mfem::Vector arb(dim); 
		arb = 0.0; 
		if (std::fabs(nor(0)) > 1e-14 or std::fabs(nor(1)) > 1e-14) {
			arb(0) = nor(1); arb(1) = -nor(0); 
		} else {
			arb(1) = nor(2); arb(2) = -nor(1); 
		}

		mfem::Vector tau1(dim), tau2(dim); 
		arb.cross3D(nor, tau1); 
		tau1.cross3D(nor, tau2); 

		double dot1 = tau1 * nor; 
		double dot2 = tau2 * nor; 
		MFEM_ASSERT(std::fabs(dot1) < 1e-14 and std::fabs(dot2) < 1e-14, "tangents not orthogonal to normal"); 
		tau.SetCol(0, tau1); 
		tau.SetCol(1, tau2); 
	}
}

int AngularQuadrature::GetReflectedAngleIndex(int angle, const mfem::Vector &nor) const 
{
	const auto &Omega = GetOmega(angle); 
	mfem::Vector rhs(dim); 
	if (dim==1) {
		rhs(0) = -Omega(0); 
	}

	else {
		mfem::DenseMatrix tau; 
		GetTangentVectors(nor, tau); 
		mfem::DenseMatrix A(dim,dim); 
		A.SetRow(0, nor); 
		rhs(0) = -(Omega*nor); 

		mfem::Vector t; 
		for (auto d=0; d<dim-1; d++) {
			tau.GetColumn(d, t); 
			A.SetRow(d+1, t); 
			rhs(d+1) = Omega * t; 
		}
		mfem::LinearSolve(A, rhs.GetData()); 		
	}
	return GetIndexForAngle(rhs); 
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