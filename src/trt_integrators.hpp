#pragma once 

#include "mfem.hpp"
#include "constants.hpp"

class BlackBodyEmissionCoefficient : public mfem::Coefficient
{
private:
	mfem::Coefficient &sigma; 
	const mfem::FiniteElement &fe; 
	const mfem::Vector &temperature; 
	mfem::Vector shape; 
public:
	BlackBodyEmissionCoefficient(mfem::Coefficient &s, const mfem::FiniteElement &el, 
		const mfem::Vector &temp)
		: sigma(s), fe(el), temperature(temp), shape(el.GetDof())
	{ }
	double Eval(mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip)
	{
		fe.CalcShape(ip, shape); 
		double T = shape * temperature; 
		return sigma.Eval(trans, ip) * constants::StefanBoltzmann * pow(T, 4); 
	}
};

class GradBlackBodyEmissionCoefficient : public mfem::Coefficient
{
private:
	mfem::Coefficient &sigma; 
	const mfem::FiniteElement &fe; 
	const mfem::Vector &temperature; 
	mfem::Vector shape; 
public:
	GradBlackBodyEmissionCoefficient(mfem::Coefficient &s, const mfem::FiniteElement &el, 
		const mfem::Vector &temp)
		: sigma(s), fe(el), temperature(temp), shape(el.GetDof())
	{ }
	double Eval(mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip)
	{
		fe.CalcShape(ip, shape); 
		double T = shape * temperature; 
		return sigma.Eval(trans, ip) * constants::StefanBoltzmann * 4.0 * pow(T, 3); 
	}	
};

class LinearCoefficientNFIntegrator : public mfem::NonlinearFormIntegrator 
{
private:
	mfem::BilinearFormIntegrator *bfi; 
public:
	LinearCoefficientNFIntegrator(mfem::BilinearFormIntegrator *bfi)
		: bfi(bfi)
	{ }
	~LinearCoefficientNFIntegrator() 
	{
		delete bfi; 
	}
	void AssembleElementVector(const mfem::FiniteElement &fe, mfem::ElementTransformation &trans, 
		const mfem::Vector &elfun, mfem::Vector &elvec)
	{
		bfi->AssembleElementVector(fe, trans, elfun, elvec); 
	}
	void AssembleElementGrad(const mfem::FiniteElement &fe, mfem::ElementTransformation &trans, 
		const mfem::Vector &elfun, mfem::DenseMatrix &elmat) 
	{
		bfi->AssembleElementMatrix(fe, trans, elmat); 
	}
};

class EnergyBalanceNonlinearFormIntegrator : public mfem::NonlinearFormIntegrator 
{
private:
	mfem::Coefficient &sigma; 
	int sigma_fe_order; 
	mfem::Vector shape; 
	mfem::IntegrationRules *rules = nullptr; 
public:
	EnergyBalanceNonlinearFormIntegrator(mfem::Coefficient &s, int sorder) 
		: sigma(s), sigma_fe_order(sorder)
	{ }
	EnergyBalanceNonlinearFormIntegrator(mfem::Coefficient &s, int sorder, int btype)
		: sigma(s), sigma_fe_order(sorder)
	{
		int type; 
		if (btype == mfem::BasisType::GaussLegendre) 
			type = mfem::Quadrature1D::GaussLegendre; 
		else if (btype == mfem::BasisType::GaussLobatto) 
			type = mfem::Quadrature1D::GaussLobatto; 
		rules = new mfem::IntegrationRules(0, type); 
	}
	~EnergyBalanceNonlinearFormIntegrator() 
	{
		if (rules) delete rules; 
	}
	void AssembleElementVector(const mfem::FiniteElement &el, mfem::ElementTransformation &trans, 
		const mfem::Vector &elfun, mfem::Vector &elvec) override;
	void AssembleElementGrad(const mfem::FiniteElement &el, mfem::ElementTransformation &trans, 
		const mfem::Vector &elfun, mfem::DenseMatrix &elmat) override;
};

