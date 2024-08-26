#pragma once 

#include "mfem.hpp"
#include "multigroup.hpp"

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
	double coef, nrho, nT; 
public:
	AnalyticGrayOpacityCoefficient(double c, double rho, double T)
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
	double coef, nrho, nT; 
	const mfem::Array<double> &energy_midpts; 
public:
	AnalyticOpacityCoefficient(double c, double rho, double T, const mfem::Array<double> &energy_midpts) 
	: coef(c), nrho(rho), nT(T), energy_midpts(energy_midpts), OpacityCoefficient(energy_midpts.Size())
	{
	}
	void Eval(mfem::Vector &v, mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) 
	{
		assert(temperature); 
		assert(density); 
		v.SetSize(vdim); 
		double temp = temperature->Eval(T, ip); 
		double rho = density->Eval(T, ip); 
		double val = coef * pow(rho, nrho) * pow(temp, nT);
		for (int g=0; g<vdim; g++) {
			const double x = energy_midpts[g] / temp;
			v(g) = val * (1.0 - exp(-x))/pow(x,3);
		}
		if (v.Min() <= 0.0) MFEM_ABORT("negative opacity");
	}
};

// helper class for representing a discrete multigroup opacity 
// on a finite element space as a grid function 
// this class ties a coefficient to its discrete representation 
// as a grid function and provides access to this data 
// as a multigrid grid function coefficient 
class ProjectedVectorCoefficient : public GridFunctionMGCoefficient 
{
private:
	mfem::VectorCoefficient &coef;

	mfem::ParGridFunction data;
public:
	ProjectedVectorCoefficient(mfem::ParFiniteElementSpace &fes, mfem::VectorCoefficient &coef)
		: coef(coef), data(&fes)
	{
		assert(coef.GetVDim() == fes.GetVDim());
		GridFunctionMGCoefficient::SetGridFunction(data);
	}
	void Project()
	{
		data.ProjectCoefficient(coef);
	}
	void Exchange()
	{
		data.ExchangeFaceNbrData();
	}
	mfem::GridFunction &GetGridFunction() { return data; }
};

// scalar version of ProjectedVectorCoefficient 
class ProjectedCoefficient : public mfem::GridFunctionCoefficient 
{
private:
	mfem::Coefficient &coef; 

	mfem::ParGridFunction data;
public:
	ProjectedCoefficient(mfem::ParFiniteElementSpace &fes, mfem::Coefficient &coef)
		: coef(coef), data(&fes), mfem::GridFunctionCoefficient(&data)
	{ }
	void Project()
	{
		data.ProjectCoefficient(coef);
	}
	void Exchange()
	{
		data.ExchangeFaceNbrData();
	}
	void SetGridFunction(const mfem::GridFunction *gf) = delete;
	mfem::GridFunction &GetGridFunction() { return data; }
};