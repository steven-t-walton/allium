#pragma once 
#include "mfem.hpp"

// enum for bitset operations to define lumping scheme 
// 0 = no lumping 
// 1 = lump mass 
// 5 = lump mass and faces 
// 7 = lump everything 
enum LumpingType {
	MASS = 1, 
	GRADIENT = 2, 
	FACE = 4 
};

static bool IsMassLumped(int lump) { return lump & LumpingType::MASS; }
static bool IsGradientLumped(int lump) { return lump & LumpingType::GRADIENT; }
static bool IsFaceLumped(int lump) { return lump & LumpingType::FACE; }

// sets up a nodal quadrature rule based on a provided finite element 
class LumpedIntegrationRule : public mfem::IntegrationRule
{
public:
	LumpedIntegrationRule(const mfem::Geometry::Type geom);
};

class QuadratureLumpedIntegrator : public mfem::BilinearFormIntegrator
{
private:
	mfem::BilinearFormIntegrator *bfi;
	int own_bfi;
public:
	QuadratureLumpedIntegrator(mfem::BilinearFormIntegrator *bfi, int own_bfi=1)
		: bfi(bfi), own_bfi(own_bfi)
	{ }
	~QuadratureLumpedIntegrator() 
	{
		if (own_bfi) delete bfi;
	}

	void AssembleElementMatrix(
		const mfem::FiniteElement &el, mfem::ElementTransformation &trans, mfem::DenseMatrix &elmat);
	void AssembleElementMatrix2(
		const mfem::FiniteElement &trial_fe, const mfem::FiniteElement &test_fe, 
		mfem::ElementTransformation &Trans, mfem::DenseMatrix &elmat);
	void AssembleFaceMatrix(
		const mfem::FiniteElement &el1,
		const mfem::FiniteElement &el2,
		mfem::FaceElementTransformations &Trans,
		mfem::DenseMatrix &elmat);
	void AssembleFaceMatrix(
		const mfem::FiniteElement &trial_fe1,
		const mfem::FiniteElement &trial_fe2,
		const mfem::FiniteElement &test_fe1,
		const mfem::FiniteElement &test_fe2,
		mfem::FaceElementTransformations &Trans,
		mfem::DenseMatrix &elmat);
};

class QuadratureLumpedNFIntegrator : public mfem::NonlinearFormIntegrator
{
private:
	mfem::NonlinearFormIntegrator *nfi;
	int own_nfi;
public:
	QuadratureLumpedNFIntegrator(mfem::NonlinearFormIntegrator *nfi, int own_nfi=1)
		: nfi(nfi), own_nfi(own_nfi)
	{ }
	~QuadratureLumpedNFIntegrator()
	{
		if (own_nfi) delete nfi;
	}

	void AssembleElementVector(
		const mfem::FiniteElement &el, mfem::ElementTransformation &trans, 
		const mfem::Vector &elfun, mfem::Vector &elvec);
	void AssembleElementGrad(
		const mfem::FiniteElement &el, mfem::ElementTransformation &trans, 
		const mfem::Vector &elfun, mfem::DenseMatrix &elmat);
};