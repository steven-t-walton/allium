#pragma once 

#include "mfem.hpp"
#include "smm_integrators.hpp"

// helper class to solve the 2x2 LDG diffusion system of the form 
// [ Mt  -DT] 
// [ D    Ma] 
class BlockDiffusionDiscretization {
protected:
	mfem::Array<int> offsets; 
	using HypreParMatrixPtr = std::unique_ptr<mfem::HypreParMatrix>; 
	HypreParMatrixPtr iMt, D, DT, Ma, S; 

	// tmp storage vectors for application of D, iMt, and DT used in EliminateRHS and BackSolve 
	mutable mfem::Vector tmp_elim_vec; 
public:

	const mfem::HypreParMatrix &SchurComplement() const { return *S; }
	// returns f -= D Mt^{-1} g
	void EliminateRHS(const mfem::Vector &g, mfem::Vector &f) const; 
	// returns Mt^{-1} (g - DT phi)
	void BackSolve(const mfem::Vector &g, const mfem::Vector &phi, mfem::Vector &J) const; 
	const mfem::Array<int> &GetOffsets() const { return offsets; }
};

class LDGDiffusionDiscretization : public BlockDiffusionDiscretization {
public:
	LDGDiffusionDiscretization(mfem::ParFiniteElementSpace &_fes, mfem::ParFiniteElementSpace &_vfes, 
		mfem::Coefficient &_total, mfem::Coefficient &_absorption, double _alpha, const mfem::Vector &beta); 
};

class IPDiffusionDiscretization : public BlockDiffusionDiscretization {
public:
	IPDiffusionDiscretization(mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &vfes, 
		mfem::Coefficient &total, mfem::Coefficient &absorption, double alpha, double kappa=-1.0); 
};

class InverseBlockDiffusionOperator : public mfem::Operator
{
private:
	const BlockDiffusionDiscretization &disc; 
	const mfem::Operator &Sinv; 

	mutable mfem::Vector phi_source; 
public:
	InverseBlockDiffusionOperator(const BlockDiffusionDiscretization &_disc, const mfem::Operator &_Sinv) 
		: disc(_disc), Sinv(_Sinv), mfem::Operator(_disc.GetOffsets().Last()) 
	{ }
	void Mult(const mfem::Vector &b, mfem::Vector &x) const; 
};

class BlockDiffusionSMMSourceOperator : public mfem::Operator 
{
private:
	mfem::ParFiniteElementSpace &fes, &vfes; 
	const AngularQuadrature &quad; 
	const TransportVectorExtents &psi_ext; 
	double alpha; 

	mfem::Array<int> offsets; 
	mfem::Vector Q0, Q1; 
public:
	BlockDiffusionSMMSourceOperator(mfem::ParFiniteElementSpace &_fes, mfem::ParFiniteElementSpace &_vfes, 
		const AngularQuadrature &_quad, const TransportVectorExtents &_psi_ext, 
		PhaseSpaceCoefficient &source_coef, PhaseSpaceCoefficient &inflow_coef, double _alpha); 
	void Mult(const mfem::Vector &psi, mfem::Vector &source) const; 
	const mfem::Array<int> &GetOffsets() const { return offsets; }
};