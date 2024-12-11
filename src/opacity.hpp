#pragma once 

#include "mfem.hpp"
#include "multigroup.hpp"
#include "brunner_opacity.hpp"

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

// evaluates Fleck and Cummings opacity 
// sigma(rho, T, E) = C rho^n1 T^n2 (1 - e^(-x))/x^3 
// where x = E/T 
// with optional "edge" 
class FleckCummingsOpacityFunction
{
private:
	double coef, nrho, nT, Emin;
	double edge_energy = std::numeric_limits<double>::max(), edge_coef = 0.0;
public:
	FleckCummingsOpacityFunction(double c, double rho, double T, double Emin)
		: coef(c), nrho(rho), nT(T), Emin(Emin)
	{ }
	void SetEdge(double E, double c) 
	{
		edge_energy = E; 
		edge_coef = c;
	}
	double operator()(double rho, double T, double E) const
	{
		const double Ehat = std::max(E, Emin);
		const double x = Ehat / T;
		double opac = coef * pow(rho, nrho) * pow(T, nT) * (1.0 - exp(-x)) / pow(Ehat,3); 
		if (Ehat > edge_energy) opac *= (1.0 + edge_coef);
		return opac;
	}
};

// evaluates Tom Brunner opacity function 
class EdgeLineOpacityFunction
{
private:
	double c0, c1, c2, Emin, Eedge, delta_s, delta_w, nT;
	int Nlines;
public:
	EdgeLineOpacityFunction(double c0, double c1, double c2, double Emin, double Eedge, 
		double delta_s, double delta_w, double nT, int Nlines)
		: c0(c0), c1(c1), c2(c2), Emin(Emin), Eedge(Eedge), delta_s(delta_s), delta_w(delta_w),
		  nT(nT), Nlines(Nlines)
	{ }
	double operator()(double rho, double T, double E) const;
};

// converts a function of (rho, T, E) to a multigroup opacity 
class MultiGroupFunctionOpacityCoefficient : public OpacityCoefficient
{
private:
	const mfem::Array<double> &bounds;
	using OpacityFunction = std::function<double(double,double,double)>; // density, temperature, energy
	OpacityFunction opacity_func;
	const mfem::IntegrationRule *rule = nullptr;
	using WeightFunction = std::function<double(double,double)>; // energy, temperature -> weight 
	WeightFunction weight_func;
public:
	MultiGroupFunctionOpacityCoefficient(const mfem::Array<double> &bounds, 
		OpacityFunction opacity_func, WeightFunction _weight_func=nullptr)
		: bounds(bounds), opacity_func(opacity_func), weight_func(_weight_func), 
		  OpacityCoefficient(bounds.Size()-1)
	{ }
	void SetIntegrationOrder(int order)
	{
		rule = &mfem::IntRules.Get(mfem::Geometry::SEGMENT, order);
	}
	void Eval(mfem::Vector &v, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) override;
	const OpacityFunction &GetOpacityFunction() const { return opacity_func; }
};

class BrunnerOpacityCoefficient : public OpacityCoefficient
{
private:
	BrunnerOpac::AnalyticEdgeOpacity *opac; 
	BrunnerOpac::MultiGroupIntegrator *integrator;

	std::vector<double> planckAvg, rossAvg, Bg, Rg;
public:
	BrunnerOpacityCoefficient(const mfem::Array<double> &bounds, double c0, double c1, 
		double c2, double Emin, double Eedge, 
		double delta_s, double delta_w, int lines);
	~BrunnerOpacityCoefficient()
	{
		delete opac; 
		delete integrator;
	}
	void Eval(mfem::Vector &v, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) override;
	double Eval(double rho, double T, double E) const 
	{
		return opac->computeSigma(E, T, rho); 
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