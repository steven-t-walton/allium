#pragma once 
#include "mfem.hpp"

// I - DL^{-1} MS 
// applies Schur complement in phi of linear transport operator 
class TransportOperator : public mfem::Operator
{
private:
	const mfem::Operator &D, &Linv, &S; 
	const mfem::Vector &source; 
	mutable mfem::Vector psi; 
public:
	TransportOperator(const mfem::Operator &_D, const mfem::Operator &_Linv, const mfem::Operator &_S, 
		const mfem::Vector &_source, mfem::Vector &_psi)
		: D(_D), Linv(_Linv), S(_S), source(_source), mfem::Operator(_D.Height(), _S.Width())
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