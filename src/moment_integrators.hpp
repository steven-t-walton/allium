#pragma once 

#include "mfem.hpp"

// implements mfem::DGDiffusionIntegrator 
// with option to allow the "Modified Interior Penalty" approach 
// where kappa_MIP = max(kappa_IP, alpha) 
class MIPDiffusionIntegrator : public mfem::BilinearFormIntegrator
{
protected:
	mfem::Coefficient *Q;
	mfem::MatrixCoefficient *MQ;
	double sigma, kappa, alpha=0.25;

	// these are not thread-safe!
	mfem::Vector shape1, shape2, dshape1dn, dshape2dn, nor, nh, ni;
	mfem::DenseMatrix jmat, dshape1, dshape2, mq, adjJ;
public:
	MIPDiffusionIntegrator(const double s, const double k)
		: Q(NULL), MQ(NULL), sigma(s), kappa(k) { }
	MIPDiffusionIntegrator(mfem::Coefficient &q, const double s, const double k, const double a)
		: Q(&q), MQ(NULL), sigma(s), kappa(k), alpha(a) { }
	MIPDiffusionIntegrator(mfem::MatrixCoefficient &q, const double s, const double k)
		: Q(NULL), MQ(&q), sigma(s), kappa(k) { }
	using BilinearFormIntegrator::AssembleFaceMatrix;
	virtual void AssembleFaceMatrix(const mfem::FiniteElement &el1,
		const mfem::FiniteElement &el2,
		mfem::FaceElementTransformations &Trans,
		mfem::DenseMatrix &elmat);
};

// integrates < {k} [u], [v] > 
class PenaltyIntegrator : public mfem::BilinearFormIntegrator
{
private:
	double kappa; 
	bool scale; 
	mfem::Coefficient *D = nullptr; 
	double limit = 0.0; 

	mfem::Vector shape1, shape2, nor; 
public:
	PenaltyIntegrator(double _kappa, bool _scale, mfem::Coefficient *d=nullptr) 
		: kappa(_kappa), scale(_scale), D(d)
	{ }
	PenaltyIntegrator(double _kappa, double _limit, mfem::Coefficient *d=nullptr) 
		: kappa(_kappa), scale(true), limit(_limit), D(d)
	{ }
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

class DGVectorJumpAverageIntegrator : public mfem::BilinearFormIntegrator
{
private:
	mfem::Vector te_shape1, te_shape2, tr_shape1, tr_shape2, nor; 
	double alpha; 
public:
	DGVectorJumpAverageIntegrator(double a=1.0) : alpha(a) { }
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