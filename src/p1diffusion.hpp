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
	double alpha; 
public:
	DGJumpAverageIntegrator(double a=1.0) : alpha(a) { } 
	using mfem::BilinearFormIntegrator::AssembleFaceMatrix; 
	void AssembleFaceMatrix(const mfem::FiniteElement &tr_fe1, const mfem::FiniteElement &tr_fe2,
		const mfem::FiniteElement &te_fe1, const mfem::FiniteElement &te_fe2, 
		mfem::FaceElementTransformations &trans, mfem::DenseMatrix &elmat); 
};

class DGVectorJumpJumpIntegrator : public mfem::BilinearFormIntegrator 
{
private:
	mfem::Vector shape1, shape2, nor; 
	double beta; 
public:
	DGVectorJumpJumpIntegrator(double b=1.0) : beta(b) { }
	using mfem::BilinearFormIntegrator::AssembleFaceMatrix; 
	void AssembleFaceMatrix(const mfem::FiniteElement &el1, const mfem::FiniteElement &el2, 
		mfem::FaceElementTransformations &trans, mfem::DenseMatrix &elmat); 
};

mfem::BlockOperator *CreateP1DiffusionDiscretization(mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &vfes, 
		mfem::Coefficient &total, mfem::Coefficient &absorption, double alpha=0.25); 

mfem::HypreParMatrix *BlockOperatorToMonolithic(const mfem::BlockOperator &bop); 

mfem::HypreParMatrix *CreateLDGDiffusionDiscretization(mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &vfes, 
	mfem::Coefficient &total, mfem::Coefficient &absorption, double alpha=0.25, mfem::Vector *beta=nullptr); 
