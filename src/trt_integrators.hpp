#pragma once 

#include "mfem.hpp"

class BlackBodyEmissionNFI : public mfem::NonlinearFormIntegrator 
{
private:
	mfem::Coefficient &sigma; 
	mfem::Vector shape; 
	int oa, ob; 
public:
	BlackBodyEmissionNFI(mfem::Coefficient &s, int a=2, int b=1) 
		: sigma(s), oa(a), ob(b)
	{ }
	void AssembleElementVector(const mfem::FiniteElement &el, mfem::ElementTransformation &trans, 
		const mfem::Vector &elfun, mfem::Vector &elvec) override;
	void AssembleElementGrad(const mfem::FiniteElement &el, mfem::ElementTransformation &trans, 
		const mfem::Vector &elfun, mfem::DenseMatrix &elmat) override;
};

class PlanckEmissionNFI : public mfem::NonlinearFormIntegrator
{
private:
	const mfem::Array<double> &group_bnds;
	mfem::VectorGridFunctionCoefficient sigma_coef;
	mfem::Vector shape, bg_shape, spectrum, sigma_g; 
	int oa, ob;
public:
	PlanckEmissionNFI(const mfem::Array<double> &group_bnds, const mfem::GridFunction &sigma_data, int a=2, int b=1)
		: group_bnds(group_bnds), oa(a), ob(b), sigma_coef(&sigma_data)
	{
		spectrum.SetSize(group_bnds.Size()-1);
		sigma_g.SetSize(group_bnds.Size()-1);
	}
	void AssembleElementVector(const mfem::FiniteElement &el, mfem::ElementTransformation &trans, 
		const mfem::Vector &elfun, mfem::Vector &elvec) override;
	void AssembleElementGrad(const mfem::FiniteElement &el, mfem::ElementTransformation &trans, 
		const mfem::Vector &elfun, mfem::DenseMatrix &elmat) override;
};

class GrayPlanckEmissionNFI : public mfem::NonlinearFormIntegrator
{
private:
	const mfem::Array<double> &group_bnds; 
	mfem::VectorGridFunctionCoefficient sigma_coef;
	mfem::Vector shape, bg_shape, spectrum, sigma_g; 
	int oa, ob; 
public:
	GrayPlanckEmissionNFI(const mfem::Array<double> &group_bnds, const mfem::GridFunction &sigma_data, int a=2, int b=1)
		: group_bnds(group_bnds), oa(a), ob(b), sigma_coef(&sigma_data)
	{
		spectrum.SetSize(group_bnds.Size()-1);
		sigma_g.SetSize(group_bnds.Size()-1);
	}
	void AssembleElementVector(const mfem::FiniteElement &el, mfem::ElementTransformation &trans, 
		const mfem::Vector &elfun, mfem::Vector &elvec) override; 
	void AssembleElementGrad(const mfem::FiniteElement &el, mfem::ElementTransformation &trans, 
		const mfem::Vector &elfun, mfem::DenseMatrix &elmat) override;
};