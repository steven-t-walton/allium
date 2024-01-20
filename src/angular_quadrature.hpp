#pragma once 
#include "mfem.hpp"

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

double ComputeAlpha(const AngularQuadrature &quad, const mfem::Vector &nor); 