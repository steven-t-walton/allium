#pragma once 
#include "mfem.hpp"
#include "tvector.hpp"
#include "angular_quadrature.hpp"

// I - DL^{-1} MS 
// applies Schur complement in phi of linear transport operator 
class TransportOperator : public mfem::Operator
{
private:
	const mfem::Operator &D, &Linv, &S; 
	mutable mfem::Vector psi; 
public:
	TransportOperator(const mfem::Operator &_D, const mfem::Operator &_Linv, const mfem::Operator &_S, 
		mfem::Vector &_psi)
		: D(_D), Linv(_Linv), S(_S), mfem::Operator(_D.Height(), _S.Width())
	{
		// get view into psi to avoid allocating psi-sized vector 
		psi.MakeRef(_psi, 0, _psi.Size()); 
	}
	void Mult(const mfem::Vector &x, mfem::Vector &y) const {
		S.Mult(x,y); 
		D.MultTranspose(y, psi); 
		Linv.Mult(psi,psi); 
		D.Mult(psi,y); 
		y -= x; 
		y *= -1.0; 
	}
};

// I + D^{-1} S preconditioner (here D is a diffusion operator)
class DiffusionSyntheticAccelerationOperator : public mfem::Solver 
{
private:
	mfem::ProductOperator op; 
public:
	DiffusionSyntheticAccelerationOperator(const mfem::Operator &Dinv, const mfem::Operator &scattering) 
		: op(&Dinv, &scattering, false, false), mfem::Solver(Dinv.Height(), false)
	{ }
	void Mult(const mfem::Vector &x, mfem::Vector &y) const 
	{
		op.Mult(x,y); 
		y += x; 
	} 

	// no op, operator assumed to be set for Dinv 
	void SetOperator(const mfem::Operator &op) { }
};

class DiscreteToMoment : public mfem::Operator {
private:
	const AngularQuadrature &quad; 
	const TransportVectorExtents &extents_psi;
	const MomentVectorExtents &extents_phi;  
public:
	DiscreteToMoment(const AngularQuadrature &_quad, 
		const TransportVectorExtents &_extents_psi, 
		const MomentVectorExtents &_extents_phi) : quad(_quad), extents_psi(_extents_psi), extents_phi(_extents_phi) {
		auto psi_size = TotalExtent(extents_psi); 
		auto phi_size = TotalExtent(extents_phi); 
		height = phi_size; 
		width = psi_size; 

		// same number of groups 
		assert(extents_psi.extent(0) == extents_phi.extent(0)); 
		// same number of space dofs 
		assert(extents_psi.extent(2) == extents_phi.extent(2)); 
	}

	void Mult(const mfem::Vector &psi, mfem::Vector &phi) const; 
	void MultTranspose(const mfem::Vector &phi, mfem::Vector &psi) const; 
};