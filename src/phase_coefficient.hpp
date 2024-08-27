#pragma once 

#include "mfem.hpp"
#include "angular_quadrature.hpp"

class PhaseSpaceCoefficient : public mfem::Coefficient 
{
protected:
	mfem::Vector Omega; 
	double energy_low, energy_high, mean_energy, energy_width; 
public:
	PhaseSpaceCoefficient() {
		Omega.SetSize(3); 
		Omega = 0.0;
	}
	double Eval(mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) override =0; 
	virtual void SetAngle(const mfem::Vector &Omega_in);
	virtual void SetEnergy(double low, double high, double mid);
};

class PWPhaseSpaceCoefficient : public PhaseSpaceCoefficient
{
protected:
	std::unordered_map<int,PhaseSpaceCoefficient*> map; 
public:
	PWPhaseSpaceCoefficient(const mfem::Array<int> &attrs, const mfem::Array<PhaseSpaceCoefficient*> &coefs); 
	void SetAngle(const mfem::Vector &Omega_in) override;
	void SetEnergy(double low, double high, double mid) override;
	double Eval(mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip); 
};

class ConstantPhaseSpaceCoefficient : public PhaseSpaceCoefficient
{
public:
	double constant; 
	ConstantPhaseSpaceCoefficient(double c) : constant(c) { }
	double Eval(mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) override {
		return constant; 
	}
};

class IsotropicGrayCoefficient : public PhaseSpaceCoefficient
{
private:
	mfem::Coefficient &coef; 
public:
	IsotropicGrayCoefficient(mfem::Coefficient &c) : coef(c) { } 
	double Eval(mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) override { return coef.Eval(trans, ip); }
};

class FunctionGrayCoefficient : public PhaseSpaceCoefficient 
{
private:
	using PhaseFunction = std::function<double(const mfem::Vector &x, const mfem::Vector &Omega)>; 
	PhaseFunction f; 
public:
	FunctionGrayCoefficient(PhaseFunction _f) : f(_f) { }
	double Eval(mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) override; 
};

class FunctionPhaseSpaceCoefficient : public PhaseSpaceCoefficient 
{
private:
	// 6D function signature 
	using PhaseFunction = std::function<double(const mfem::Vector &x, const mfem::Vector &Omega, double E)>; 
	// 7D function signature 
	using PhaseFunctionTD = std::function<double(const mfem::Vector &x, const mfem::Vector &Omega, double E, double time)>; 
	PhaseFunction f; 
	PhaseFunctionTD ftd; 
public:
	FunctionPhaseSpaceCoefficient(PhaseFunction F) : f(std::move(F)) { }
	FunctionPhaseSpaceCoefficient(PhaseFunctionTD Ftd) : ftd(std::move(Ftd)) { }
	double Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override; 
};

// evaluates multigroup planck emission 
// designed for computing initial conditions and boundary conditions 
// where a scalar coefficient is needed for each angle and energy group 
class PlanckEmissionPSCoefficient : public PhaseSpaceCoefficient
{
private:
	mfem::Coefficient &temperature;
public:
	PlanckEmissionPSCoefficient(mfem::Coefficient &T) : temperature(T)
	{ }
	double Eval(mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) override;
};