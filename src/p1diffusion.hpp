#pragma once 

#include "mfem.hpp"

class PenaltyIntegrator : public mfem::BilinearFormIntegrator
{
private:
	double kappa; 
	bool scale; 

	mfem::Vector shape1, shape2, nor; 
public:
	PenaltyIntegrator(double _kappa, bool _scale) : kappa(_kappa), scale(_scale) {
	}
	using mfem::BilinearFormIntegrator::AssembleFaceMatrix; 
	void AssembleFaceMatrix(const mfem::FiniteElement &el1, const mfem::FiniteElement &el2, 
		mfem::FaceElementTransformations &trans, mfem::DenseMatrix &elmat); 
};

class DGJumpAverageIntegrator : public mfem::BilinearFormIntegrator
{
private:
	mfem::Vector te_shape1, te_shape2, tr_shape1, tr_shape2, nor; 
	double alpha = 1.0; 
public:
	DGJumpAverageIntegrator() { }
	DGJumpAverageIntegrator(double a) : alpha(a) { } 
	using mfem::BilinearFormIntegrator::AssembleFaceMatrix; 
	void AssembleFaceMatrix(const mfem::FiniteElement &tr_fe1, const mfem::FiniteElement &tr_fe2,
		const mfem::FiniteElement &te_fe1, const mfem::FiniteElement &te_fe2, 
		mfem::FaceElementTransformations &trans, mfem::DenseMatrix &elmat); 
};

class DGVectorJumpJumpIntegrator : public mfem::BilinearFormIntegrator 
{
private:
	mfem::Vector shape1, shape2, nor; 
public:
	using mfem::BilinearFormIntegrator::AssembleFaceMatrix; 
	void AssembleFaceMatrix(const mfem::FiniteElement &el1, const mfem::FiniteElement &el2, 
		mfem::FaceElementTransformations &trans, mfem::DenseMatrix &elmat); 
};

class BoundaryNormalFaceLFIntegrator : public mfem::LinearFormIntegrator 
{
private:
	mfem::Vector shape, nor; 
	mfem::Coefficient &inflow; 
	int oa, ob; 
public:
	BoundaryNormalFaceLFIntegrator(mfem::Coefficient &_inflow, int a=1, int b=1) : inflow(_inflow), oa(a), ob(b) { } 
	void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvec) {
		MFEM_ABORT("only call for faces"); 
	}
	void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::FaceElementTransformations &trans, mfem::Vector &elvec); 
};