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

