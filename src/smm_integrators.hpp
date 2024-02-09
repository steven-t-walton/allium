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

class CSMMZerothMomentFaceLFIntegrator : public mfem::LinearFormIntegrator
{
private:
	const mfem::ParGridFunction &beta; 
	int oa, ob; 

	mfem::Array<int> beta_dofs1, beta_dofs2; 
	mfem::Vector shape1, shape2, tr_shape1, tr_shape2, beta_all1, beta_all2, beta_trace1, beta_trace2; 
public:
	CSMMZerothMomentFaceLFIntegrator(const mfem::ParGridFunction &_beta, int a=2, int b=1) 
		: beta(_beta), oa(a), ob(b) 
	{ }
	void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvec) {
		MFEM_ABORT("only call for faces"); 
	}
	void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::FaceElementTransformations &trans, mfem::Vector &elvec); 	
	void AssembleRHSElementVect(const mfem::FiniteElement &el1, const mfem::FiniteElement &el2, 
		mfem::FaceElementTransformations &trans, mfem::Vector &elvec); 
};

class CSMMFirstMomentFaceLFIntegrator : public mfem::LinearFormIntegrator
{
private:
	const mfem::ParGridFunction &tensor; 
	int oa, ob; 

	mfem::Array<int> tr_vdofs1, tr_vdofs2; 
	mfem::Vector shape1, shape2, tr_shape1, tr_shape2; 
	mfem::Vector tensor_all1, tensor_all2, tensor_tr1, tensor_tr2; 
public:
	CSMMFirstMomentFaceLFIntegrator(const mfem::ParGridFunction &_tensor, int a=2, int b=1)
		: tensor(_tensor), oa(a), ob(b)
	{ }
	void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvec) {
		MFEM_ABORT("only call for faces"); 
	}
	void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::FaceElementTransformations &trans, mfem::Vector &elvec); 	
	void AssembleRHSElementVect(const mfem::FiniteElement &el1, const mfem::FiniteElement &el2, 
		mfem::FaceElementTransformations &trans, mfem::Vector &elvec); 	
};

class LDGTraceIntegrator : public mfem::BilinearFormIntegrator
{
protected:
	const mfem::Vector *beta = nullptr;
	mfem::Coefficient *coef = nullptr; 

	mfem::Vector tr_shape1, tr_shape2, te_shape1, te_shape2; 
	mfem::Vector nor; 
	mfem::DenseMatrix A11, A12, A21, A22; 
public:
	LDGTraceIntegrator(const mfem::Vector *b=nullptr) { beta = b; }
	LDGTraceIntegrator(mfem::Coefficient &c, const mfem::Vector *b=nullptr) 
		: coef(&c), beta(b) 
	{ }
	void AssembleFaceMatrix(
		const mfem::FiniteElement &tr_fe1,
		const mfem::FiniteElement &tr_fe2,
		const mfem::FiniteElement &te_fe1, 
		const mfem::FiniteElement &te_fe2,
		mfem::FaceElementTransformations &T, 
		mfem::DenseMatrix &elmat);
};