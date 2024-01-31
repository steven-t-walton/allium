#pragma once 
#include "mfem.hpp"
#include "tvector.hpp"
#include "angular_quadrature.hpp"

class WeakTensorDivergenceLFIntegrator : public mfem::LinearFormIntegrator 
{
private:
	mfem::MatrixCoefficient &Tcoef; 
	int oa, ob; 

	mfem::DenseMatrix gshape, T; 
	mfem::Vector T_flat, gradT; 
public:
	WeakTensorDivergenceLFIntegrator(mfem::MatrixCoefficient &_T, int a=2, int b=1) : Tcoef(_T), oa(a), ob(b) { }
	void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &trans, mfem::Vector &elvec); 
};

class VectorJumpTensorAverageLFIntegrator : public mfem::LinearFormIntegrator 
{
private:
	mfem::MatrixCoefficient &Tcoef; 
	int oa, ob; 

	mfem::Vector shape1, shape2, nor, Tn1, Tn2; 
	mfem::DenseMatrix T; 
public:
	VectorJumpTensorAverageLFIntegrator(mfem::MatrixCoefficient &_T, int a=2, int b=1) : Tcoef(_T), oa(a), ob(b) { }
	void AssembleRHSElementVect(const mfem::FiniteElement&, mfem::ElementTransformation&, mfem::Vector&) {
		MFEM_ABORT("call on faces only"); 
	}
	void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::FaceElementTransformations &trans, 
		mfem::Vector &elvec); 
	void AssembleRHSElementVect(const mfem::FiniteElement &el1, const mfem::FiniteElement &el2, 
		mfem::FaceElementTransformations &trans, mfem::Vector &elvec); 
};

class BoundaryNormalFaceLFIntegrator : public mfem::LinearFormIntegrator 
{
private:
	mfem::Vector shape, nor; 
	mfem::Coefficient &inflow; 
	int oa, ob; 
public:
	BoundaryNormalFaceLFIntegrator(mfem::Coefficient &_inflow, int a=2, int b=1) : inflow(_inflow), oa(a), ob(b) { } 
	void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvec) {
		MFEM_ABORT("only call for faces"); 
	}
	void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::FaceElementTransformations &trans, mfem::Vector &elvec); 
};

class ProjectedCoefBoundaryLFIntegrator : public mfem::LinearFormIntegrator
{
private:
	mfem::Coefficient &Q;
	const mfem::FiniteElementCollection &fec;  
	int oa, ob; 

	mfem::Vector shape, Qnodes, tr_shape; 
public:
	ProjectedCoefBoundaryLFIntegrator(mfem::Coefficient &q, const mfem::FiniteElementCollection &_fec, int a=1, int b=1) 
		: Q(q), fec(_fec), oa(a), ob(b)
	{ }
	void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvec) {
		MFEM_ABORT("only call for faces"); 
	}
	void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::FaceElementTransformations &trans, mfem::Vector &elvec); 
};

class ProjectedCoefBoundaryNormalLFIntegrator : public mfem::LinearFormIntegrator
{
private:
	mfem::Coefficient &Q; 
	const mfem::FiniteElementCollection &fec; 
	int oa, ob; 

	mfem::Vector shape, nor, Qnodes, tr_shape; 
public:
	ProjectedCoefBoundaryNormalLFIntegrator(mfem::Coefficient &q, const mfem::FiniteElementCollection &_fec, int a=1, int b=1) 
		: Q(q), fec(_fec), oa(a), ob(b)
	{ }
	void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvec) {
		MFEM_ABORT("only call for faces"); 
	}
	void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::FaceElementTransformations &trans, mfem::Vector &elvec); 	
};

class SMMCorrectionTensorCoefficient : public mfem::MatrixArrayCoefficient
{
private:
	mfem::ParFiniteElementSpace &fes; 
	const AngularQuadrature &quad; 
	ConstTransportVectorView psi; 
public:
	mfem::Array<mfem::ParGridFunction*> gfs; 

	SMMCorrectionTensorCoefficient(mfem::ParFiniteElementSpace &_fes, const AngularQuadrature &_quad, ConstTransportVectorView _psi);
	~SMMCorrectionTensorCoefficient(); 
};

class SMMBdrCorrectionFactorCoefficient : public mfem::Coefficient 
{
private:
	mfem::ParFiniteElementSpace &fes; 
	const AngularQuadrature &quad; 
	ConstTransportVectorView psi; 
	double alpha; 

	int dim; 
	mfem::Vector nor, shape; 
public:
	SMMBdrCorrectionFactorCoefficient(mfem::ParFiniteElementSpace &_fes, const AngularQuadrature &_quad, 
		ConstTransportVectorView _psi, double _alpha=0.5); 
	double Eval(mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip); 
};