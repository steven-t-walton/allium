#pragma once 
#include "mfem.hpp"
#include "tvector.hpp"

class AngularQuadrature {
protected:
	std::vector<mfem::Vector> Omegas; 
	std::vector<double> weights; 
	double weights_sum = 0.0; 
public:
	int Size() const { return weights.size(); }
	const mfem::Vector &GetOmega(int angle) const { return Omegas[angle]; }
	double GetWeight(int angle) const { return weights[angle]; }
	double SumWeights() const { return weights_sum; }
};

class LevelSymmetricQuadrature : public AngularQuadrature {
private:
	int dim; 
public:
	LevelSymmetricQuadrature(int _order, int _dim); 
};

class SingleAngleQuadratureRule : public AngularQuadrature {
public:
	SingleAngleQuadratureRule(const AngularQuadrature &rule, int angle) {
		Omegas.resize(1); 
		weights.resize(1); 
		weights[0] = 4*M_PI; 
		weights_sum = 4*M_PI; 
		Omegas[0] = rule.GetOmega(angle); 
	}
};

class GaussLegendreQuadratureRule : public AngularQuadrature {
public:
	GaussLegendreQuadratureRule(int order, int dim); 
};

class DiscreteToMoment : public mfem::Operator {
private:
	const AngularQuadrature &quad; 
	const TransportVectorExtents &extents_psi;
	const MomentVectorExtents &extents_phi;  
public:
	DiscreteToMoment(const AngularQuadrature &_quad, 
		const TransportVectorExtents &_extents_psi, 
		const MomentVectorExtents &_extents_phi) : quad(_quad), extents_psi(_extents_psi), extents_phi(_extents_phi) {
		auto psi_size = TotalExtent(extents_psi); 
		auto phi_size = TotalExtent(extents_phi); 
		height = phi_size; 
		width = psi_size; 

		// same number of groups 
		assert(extents_psi.extent(0) == extents_phi.extent(1)); 
		// same number of space dofs 
		assert(extents_psi.extent(2) == extents_phi.extent(2)); 
	}

	void Mult(const mfem::Vector &psi, mfem::Vector &phi) const; 
	void MultTranspose(const mfem::Vector &phi, mfem::Vector &psi) const; 
};

double ComputeAlpha(const AngularQuadrature &quad, const mfem::Vector &nor); 