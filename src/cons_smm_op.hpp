#pragma once 
#include "mfem.hpp"
#include "angular_quadrature.hpp"
#include "tvector.hpp"
#include "dg_trace_coll.hpp"

// make test solving P1 diffusion with this source operator! 
class ConsistentSMMSourceOperator : public mfem::Operator {
private:
	mfem::ParFiniteElementSpace &fes, &vfes; 
	const AngularQuadrature &quad; 
	const TransportVectorExtents &psi_ext; 
	double alpha; 

	std::unique_ptr<const mfem::Table> element_to_face; 

	mfem::Array<int> offsets; 
	mfem::Vector Q0, Q1; 
	std::unique_ptr<DGTrace_FECollection> trace_coll; 
	std::unique_ptr<mfem::ParFiniteElementSpace> trace_fes, trace_vfes; 
	mutable mfem::ParGridFunction beta, tensor; 
public:
	ConsistentSMMSourceOperator(mfem::ParFiniteElementSpace &_fes, mfem::ParFiniteElementSpace &_vfes, 
		const AngularQuadrature &_quad, const TransportVectorExtents &_psi_ext, ConstTransportVectorView source_vec, double _alpha); 
	void Mult(const mfem::Vector &psi, mfem::Vector &source) const; 

	// change to tensor closure only for first moment 
	// hat(P)n - n/3 {{ phi }} - n/6/alpha [[ J.n ]]
	// An = Pn^+ - n/6 phi - n/6/alpha J.n 
	// -A(-n) = Pn^- - n/6 phi + n/6/alpha J.n
	// closure is An1 - A(-n)2 (always compute outflow normal component of pressure)
	void ProjectClosuresToFaces(ConstTransportVectorView psi, mfem::ParGridFunction &beta, mfem::ParGridFunction &tensor) const; 
};

class CSMMFaceIntegrator0: public mfem::LinearFormIntegrator
{
private:
	ConstTransportVectorView psi; 
	mfem::ParFiniteElementSpace &fes; 
	const AngularQuadrature &quad; 
	double alpha; 
	int oa, ob; 

	mfem::Vector shape1, shape2, nor; 
public:
	CSMMFaceIntegrator0(mfem::ParFiniteElementSpace &_fes, const AngularQuadrature &_quad, ConstTransportVectorView _psi, 
		double _alpha, int a=2, int b=1)
		: fes(_fes), quad(_quad), psi(_psi), alpha(_alpha), oa(a), ob(b) 
	{ }
	void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvec) {
		MFEM_ABORT("only call for faces"); 
	}
	void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::FaceElementTransformations &trans, mfem::Vector &elvec); 	
	void AssembleRHSElementVect(const mfem::FiniteElement &el1, const mfem::FiniteElement &el2, 
		mfem::FaceElementTransformations &trans, mfem::Vector &elvec); 	
};

class CSMMFaceIntegrator1 : public mfem::LinearFormIntegrator 
{
private:
	ConstTransportVectorView psi; 
	mfem::ParFiniteElementSpace &fes; 
	const AngularQuadrature &quad; 
	double alpha; 
	int oa, ob; 

	mfem::Vector shape1, shape2, nor; 
public:
	CSMMFaceIntegrator1(mfem::ParFiniteElementSpace &_fes, const AngularQuadrature &_quad, ConstTransportVectorView _psi, 
		double _alpha, int a=2, int b=1)
		: fes(_fes), quad(_quad), psi(_psi), alpha(_alpha), oa(a), ob(b) 
	{ }
	void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvec) {
		MFEM_ABORT("only call for faces"); 
	}
	void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::FaceElementTransformations &trans, mfem::Vector &elvec); 	
	void AssembleRHSElementVect(const mfem::FiniteElement &el1, const mfem::FiniteElement &el2, 
		mfem::FaceElementTransformations &trans, mfem::Vector &elvec); 	
};