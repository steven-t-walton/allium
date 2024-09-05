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

class VectorFEAvgTensorJumpLFIntegrator : public mfem::LinearFormIntegrator
{
private:
	mfem::MatrixCoefficient &Tcoef;
	int oa, ob;

	mfem::Vector nor, Tn1, Tn2;
	mfem::DenseMatrix T, vshape1, vshape2;
public:
	VectorFEAvgTensorJumpLFIntegrator(mfem::MatrixCoefficient &T, int a=2, int b=1)
		: Tcoef(T), oa(a), ob(b)
	{ }
	void AssembleRHSElementVect(const mfem::FiniteElement&, mfem::ElementTransformation&, mfem::Vector&) override
	{
		MFEM_ABORT("call on interior faces only"); 
	}
	void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::FaceElementTransformations &trans, 
		mfem::Vector &elvec) override
	{
		MFEM_ABORT("call on interior faces only");
	}
	void AssembleRHSElementVect(const mfem::FiniteElement &el1, const mfem::FiniteElement &el2, 
		mfem::FaceElementTransformations &trans, mfem::Vector &elvec) override;
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

class MixedVectorScalarMassIntegrator : public mfem::BilinearFormIntegrator
{
private:
	mfem::VectorCoefficient &coef;

	mfem::Vector shape, coef_eval;
	mfem::DenseMatrix shape_vvt;
public:
	MixedVectorScalarMassIntegrator(mfem::VectorCoefficient &coef)
		: coef(coef)
	{
		coef_eval.SetSize(coef.GetVDim());
	}
	void AssembleElementMatrix2(const mfem::FiniteElement &trial_fe, const mfem::FiniteElement &test_fe, 
		mfem::ElementTransformation &trans, mfem::DenseMatrix &elmat) override;
};

class GradAverageTensorJumpLFIntegrator : public mfem::LinearFormIntegrator
{
private:
	mfem::Coefficient &total_coef;
	mfem::MatrixCoefficient &Tcoef;
	int oa, ob;

	mfem::DenseMatrix T, gshape1, gshape2;
	mfem::Vector Tn1, Tn2, nor;
public:
	GradAverageTensorJumpLFIntegrator(mfem::Coefficient &total_coef, mfem::MatrixCoefficient &Tcoef, 
		int oa=2, int ob=1)
		: total_coef(total_coef), Tcoef(Tcoef), oa(oa), ob(ob)
	{ }
	void AssembleRHSElementVect(const mfem::FiniteElement&, mfem::ElementTransformation&, mfem::Vector&)
	{
		MFEM_ABORT("call on interior faces only"); 
	}
	void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::FaceElementTransformations &trans, 
		mfem::Vector &elvec)
	{
		MFEM_ABORT("call on interior faces only");
	}
	void AssembleRHSElementVect(const mfem::FiniteElement &el1, const mfem::FiniteElement &el2, 
		mfem::FaceElementTransformations &trans, mfem::Vector &elvec); 	
};