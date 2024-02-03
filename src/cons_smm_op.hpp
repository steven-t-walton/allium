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
	const mfem::Array<int> &Offsets() const { return offsets; }
};

class ConsistentLDGSMMSourceOperator : public mfem::Operator {
private:
	mfem::ParFiniteElementSpace &fes, &vfes; 
	const AngularQuadrature &quad; 
	const TransportVectorExtents &psi_ext; 
	double alpha; 	

	ConsistentSMMSourceOperator base_source_op; 
	MomentVectorExtents phi_ext; 
	mutable mfem::Vector moments;

	using HypreParMatrixPtr = std::unique_ptr<mfem::HypreParMatrix>; 
	HypreParMatrixPtr F1, F2;  
public:
	ConsistentLDGSMMSourceOperator(mfem::ParFiniteElementSpace &_fes, mfem::ParFiniteElementSpace &_vfes, 
		const AngularQuadrature &quad, const TransportVectorExtents &_psi_ext, ConstTransportVectorView source_vec, 
		double _alpha, const mfem::Vector &beta); 
	void Mult(const mfem::Vector &psi, mfem::Vector &source) const; 
};

// hat(P)n - n/3 {{ phi }} - n/6/alpha [[ J.n ]]
// An = Pn^+ - n/6 phi - n/6/alpha J.n 
// -A(-n) = Pn^- - n/6 phi + n/6/alpha J.n
// closure is An1 - A(-n)2 (always compute outflow normal component of pressure)
void ProjectClosuresToFaces(const mfem::ParFiniteElementSpace &fes, const AngularQuadrature &quad, ConstTransportVectorView psi, 
	double alpha, mfem::ParGridFunction &beta, mfem::ParGridFunction &tensor); 