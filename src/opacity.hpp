#pragma once 

#include "mfem.hpp"

class OpacityCoefficient : public mfem::VectorCoefficient {
protected:
	mfem::Coefficient *temperature = nullptr, *density = nullptr; 
public:
	OpacityCoefficient(int G) : mfem::VectorCoefficient(G) { } 
	virtual void SetTemperature(mfem::Coefficient &T) {
		temperature = &T; 
	}
	virtual void SetDensity(mfem::Coefficient &rho) {
		density = &rho; 
	}
};

class PWOpacityCoefficient : public OpacityCoefficient {
private:
	std::unordered_map<int,OpacityCoefficient*> map; 
public:
	PWOpacityCoefficient(const mfem::Array<int> &attrs, const mfem::Array<OpacityCoefficient*> &coefs); 
	void SetTemperature(mfem::Coefficient &T) override; 
	void SetDensity(mfem::Coefficient &rho) override; 

	void Eval(mfem::Vector &v, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip); 
};

class ConstantOpacityCoefficient : public OpacityCoefficient {
private:
	mfem::Vector constants; 
public:
	ConstantOpacityCoefficient(const mfem::Vector &c) : constants(c), OpacityCoefficient(c.Size()) { }
	void Eval(mfem::Vector &v, mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
	{
		v = constants; 
	}
};

class AnalyticGrayOpacityCoefficient : public OpacityCoefficient {
private:
	double coef; 
	int nrho, nT; 
public:
	AnalyticGrayOpacityCoefficient(double c, int rho, int T)
	: coef(c), nrho(rho), nT(T), OpacityCoefficient(1)
	{
	}

	void Eval(mfem::Vector &v, mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) {
		assert(temperature);
		assert(density); 
		v.SetSize(1); 
		v(0) = coef * pow(density->Eval(T, ip), nrho) * pow(temperature->Eval(T, ip), nT);  
		if (v(0) < 0) MFEM_ABORT("negative opacity"); 
	}
};

class AnalyticOpacityCoefficient : public OpacityCoefficient {
private:
	double coef; 
	const mfem::Array<double> &energy_grid; 
public:
	AnalyticOpacityCoefficient(double c, const mfem::Array<double> &grid) 
	: coef(c), energy_grid(grid), OpacityCoefficient(grid.Size()-1)
	{
	}
	void Eval(mfem::Vector &v, mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) 
	{
		assert(temperature); 
		assert(density); 
		v.SetSize(vdim); 
		double temp = temperature->Eval(T, ip); 
		double rho = density->Eval(T, ip); 
		for (int g=0; g<vdim; g++) {
			// energy grid cell center / T 
			double u = (energy_grid[g+1] + energy_grid[g])/2 / temp;  
			v(g) = rho*coef/pow(u,3) * (1.0 - exp(-u))/pow(temp, 3);
		}
	}
};

class GroupCollapseOperator : public mfem::Operator {
private:
	mfem::FiniteElementSpace &fes; 
	const mfem::Array<double> &energy_grid; 
	mfem::VectorCoefficient *f; 
public:
	GroupCollapseOperator(mfem::FiniteElementSpace &sigma_space, const mfem::Array<double> &grid, 
		mfem::VectorCoefficient *weight_func=nullptr) 
	: fes(sigma_space), energy_grid(grid), f(weight_func) 
	{
		width = fes.GetVSize(); // G x space dofs 
		height = fes.GetNDofs(); // 1 x space dofs 
		assert(grid.Size() - 1 == fes.GetVDim()); 
	}

	void Mult(const mfem::Vector &sigma_mf, mfem::Vector &sigma_gray) const; 
};