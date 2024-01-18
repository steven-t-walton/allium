#pragma once 
#include "mfem.hpp"

class WeakTensorDivergenceLFIntegrator : public mfem::LinearFormIntegrator 
{
private:
	mfem::MatrixCoefficient &Tcoef; 
	int oa, ob; 

	mfem::DenseMatrix gshape, T; 
	mfem::Vector T_flat, gradT; 
public:
	WeakTensorDivergenceLFIntegrator(mfem::MatrixCoefficient &_T, int a=2, int b=0) : Tcoef(_T), oa(a), ob(b) { }
	void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &trans, mfem::Vector &elvec); 
};