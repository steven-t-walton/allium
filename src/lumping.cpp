#include "lumping.hpp"

LumpedIntegrationRule::LumpedIntegrationRule(const mfem::Geometry::Type geom)
{
	const auto *vertices = mfem::Geometries.GetVertices(geom); 
	SetSize(vertices->GetNPoints()); 
	const double w = mfem::Geometries.Volume[geom]/Size(); 
	for (int i=0; i<Size(); i++) {
		(*this)[i] = vertices->IntPoint(i); 
		(*this)[i].weight = w;
	}
}

void QuadratureLumpedIntegrator::AssembleElementMatrix(
	const mfem::FiniteElement &el, mfem::ElementTransformation &trans, mfem::DenseMatrix &elmat)
{
	LumpedIntegrationRule rule(trans.GetGeometryType()); 
	bfi->SetIntegrationRule(rule); 
	bfi->AssembleElementMatrix(el, trans, elmat); 
}

void QuadratureLumpedIntegrator::AssembleElementMatrix2(
	const mfem::FiniteElement &trial_fe, const mfem::FiniteElement &test_fe, 
	mfem::ElementTransformation &Trans, mfem::DenseMatrix &elmat)
{
	assert(trial_fe.GetDof() == test_fe.GetDof()); 
	LumpedIntegrationRule rule(Trans.GetGeometryType()); 
	bfi->SetIntegrationRule(rule); 
	bfi->AssembleElementMatrix2(trial_fe, test_fe, Trans, elmat); 
}

void QuadratureLumpedIntegrator::AssembleFaceMatrix(
	const mfem::FiniteElement &el1,
	const mfem::FiniteElement &el2,
	mfem::FaceElementTransformations &Trans,
	mfem::DenseMatrix &elmat)
{
	LumpedIntegrationRule rule(Trans.GetGeometryType()); 
	bfi->SetIntegrationRule(rule); 
	bfi->AssembleFaceMatrix(el1, el2, Trans, elmat); 
}

void QuadratureLumpedIntegrator::AssembleFaceMatrix(
	const mfem::FiniteElement &trial_fe1,
	const mfem::FiniteElement &trial_fe2,
	const mfem::FiniteElement &test_fe1,
	const mfem::FiniteElement &test_fe2,
	mfem::FaceElementTransformations &Trans,
	mfem::DenseMatrix &elmat)
{
	LumpedIntegrationRule rule(Trans.GetGeometryType()); 
	bfi->SetIntegrationRule(rule); 
	bfi->AssembleFaceMatrix(trial_fe1, trial_fe2, test_fe1, test_fe2, Trans, elmat); 
}

void QuadratureLumpedNFIntegrator::AssembleElementVector(
	const mfem::FiniteElement &el, mfem::ElementTransformation &trans, 
	const mfem::Vector &elfun, mfem::Vector &elvec)
{
	LumpedIntegrationRule rule(trans.GetGeometryType());
	nfi->SetIntegrationRule(rule);
	nfi->AssembleElementVector(el, trans, elfun, elvec);
}

void QuadratureLumpedNFIntegrator::AssembleElementGrad(
	const mfem::FiniteElement &el, mfem::ElementTransformation &trans, 
	const mfem::Vector &elfun, mfem::DenseMatrix &elmat)
{
	LumpedIntegrationRule rule(trans.GetGeometryType());
	nfi->SetIntegrationRule(rule);
	nfi->AssembleElementGrad(el, trans, elfun, elmat);
}

void QuadratureLumpedLFIntegrator::AssembleRHSElementVect(const mfem::FiniteElement &el, 
	mfem::ElementTransformation &Tr, mfem::Vector &elvect)
{
	LumpedIntegrationRule rule(Tr.GetGeometryType());
	lfi->SetIntRule(&rule);
	lfi->AssembleRHSElementVect(el, Tr, elvect);
}
void QuadratureLumpedLFIntegrator::AssembleRHSElementVect(const mfem::FiniteElement &el, 
	mfem::FaceElementTransformations &Tr, mfem::Vector &elvect)
{
	LumpedIntegrationRule rule(Tr.GetGeometryType());
	lfi->SetIntRule(&rule);
	lfi->AssembleRHSElementVect(el, Tr, elvect);
}
void QuadratureLumpedLFIntegrator::AssembleRHSElementVect(const mfem::FiniteElement &el1, const mfem::FiniteElement &el2, 
	mfem::FaceElementTransformations &Tr, mfem::Vector &elvect)
{
	LumpedIntegrationRule rule(Tr.GetGeometryType());
	lfi->SetIntRule(&rule);
	lfi->AssembleRHSElementVect(el1, el2, Tr, elvect);
}

void QuadratureLumpedMGIntegrator::AssembleElementMatrices(
	const mfem::FiniteElement &fe, mfem::ElementTransformation &trans, const mfem::Array2D<mfem::DenseMatrix*> &elmats)
{
	LumpedIntegrationRule rule(trans.GetGeometryType());
	bfi->SetIntegrationRule(rule);
	bfi->AssembleElementMatrices(fe, trans, elmats);
}