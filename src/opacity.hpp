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
	double Emin;
public:
	AnalyticOpacityCoefficient(double c, double rho, double T, 
		const mfem::Array<double> &energy_midpts, double Emin=0.0) 
		: coef(c), nrho(rho), nT(T), energy_midpts(energy_midpts), Emin(Emin),
		  OpacityCoefficient(energy_midpts.Size())
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
			const double Ehat = std::max(Emin, energy_midpts[g]);
			const double x = Ehat / temp;
			v(g) = val * (1.0 - exp(-x))/pow(x,3);
		}
		if (v.Min() <= 0.0) MFEM_ABORT("negative opacity");
	}
};

class AnalyticEdgeOpacityCoefficient : public OpacityCoefficient {
private:
	double c0, c1, c2, Emin, Eedge, delta_s, delta_w; 
	int Nlines;
	const mfem::Array<double> &energy_bounds;
	int int_order;
public:
	AnalyticEdgeOpacityCoefficient(double c0, double c1, double c2, 
		double Emin, double Eedge, double delta_s, double delta_w, int Nlines, 
		const mfem::Array<double> &energy_bounds, int int_order=1)
		: c0(c0), c1(c1), c2(c2), Emin(Emin), Eedge(Eedge), 
		  delta_s(delta_s), delta_w(delta_w), Nlines(Nlines), 
		  energy_bounds(energy_bounds), int_order(int_order), 
		  OpacityCoefficient(energy_bounds.Size()-1)
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
			v(g) = IntegrateOpacity(energy_bounds[g], energy_bounds[g+1], temp, rho);
		}
		if (v.Min() <= 0.0) MFEM_ABORT("negative opacity");
	}
	double ComputeOpacity(const double T, const double rho, const double E); 
	double IntegrateOpacity(const double Elow, const double Ehigh, const double T, const double rho);
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