#pragma once 

#include "mfem.hpp"
#include "angular_quadrature.hpp"

class PhaseSpaceCoefficient : public mfem::Coefficient 
{
protected:
	mfem::Vector Omega; 
	double energy; 
public:
	PhaseSpaceCoefficient() {
		Omega.SetSize(3); 
		Omega = 0.0;
	}
	double Eval(mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) = 0; 
	virtual void SetAngle(const mfem::Vector &Omega_in) {
		// copy Omega preserving Omega.Size() = 3 
		for (int d=0; d<Omega_in.Size(); d++) {
			Omega(d) = Omega_in(d); 
		}		
	}
	virtual void SetEnergy(double E) {
		energy = E; 
	}
};

class PWPhaseSpaceCoefficient : public PhaseSpaceCoefficient
{
protected:
	std::unordered_map<int,PhaseSpaceCoefficient*> map; 
public:
	PWPhaseSpaceCoefficient(const mfem::Array<int> &attrs, const mfem::Array<PhaseSpaceCoefficient*> &coefs); 
	void SetAngle(const mfem::Vector &Omega_in);
	void SetEnergy(double E);  
	double Eval(mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip); 
};

class ConstantPhaseSpaceCoefficient : public PhaseSpaceCoefficient
{
public:
	double constant; 
	ConstantPhaseSpaceCoefficient(double c) : constant(c) { }
	double Eval(mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) {
		return constant; 
	}
};

class IsotropicGrayCoefficient : public PhaseSpaceCoefficient
{
private:
	mfem::Coefficient &coef; 
public:
	IsotropicGrayCoefficient(mfem::Coefficient &c) : coef(c) { } 
	double Eval(mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) { return coef.Eval(trans, ip); }
};

class FunctionGrayCoefficient : public PhaseSpaceCoefficient 
{
private:
	using PhaseFunction = std::function<double(const mfem::Vector &x, const mfem::Vector &Omega)>; 
	PhaseFunction f; 
public:
	FunctionGrayCoefficient(PhaseFunction _f) : f(_f) { }
	double Eval(mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip); 
};

class InflowPartialCurrentCoefficient : public mfem::Coefficient 
{
private:
	PhaseSpaceCoefficient &phase_coef; 
	const AngularQuadrature &quad; 
	double scale; 

	mfem::Vector nor; 
public:
	InflowPartialCurrentCoefficient(PhaseSpaceCoefficient &pc, const AngularQuadrature &q, double s=1.0) 
		: phase_coef(pc), quad(q), scale(s)
	{ }

	double Eval(mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip); 
};
