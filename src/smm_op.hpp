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
		mfem::Coefficient &_total, mfem::Coefficient &_absorption, double _alpha, const mfem::Vector &beta, 
		bool scale_ldg_stabilization=false, int reflection_bdr_attr=-1); 
};

class IPDiffusionDiscretization : public BlockDiffusionDiscretization {
public:
	IPDiffusionDiscretization(mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &vfes, 
		mfem::Coefficient &total, mfem::Coefficient &absorption, double alpha, double kappa=-1.0, 
		bool mip=false, bool scale_ip_stabilization=true, int reflection_bdr_attr=-1); 
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

class MomentMethodFixedPointOperator : public mfem::Operator {
private:
	const mfem::Operator &D, &Linv, &S, &moment; 
	const mfem::Vector &source;  
	mutable mfem::Vector psi;
	mutable mfem::StopWatch total_timer, sweep_timer, moment_timer; 
public:
	MomentMethodFixedPointOperator(const mfem::Operator &_D, const mfem::Operator &_Linv, 
		const mfem::Operator &_S, const mfem::Operator &_moment, const mfem::Vector &_source, mfem::Vector &_psi)
		: D(_D), Linv(_Linv), S(_S), moment(_moment), source(_source), mfem::Operator(_moment.Height())
	{
		psi.MakeRef(_psi, 0, _psi.Size()); 
	}
	void Mult(const mfem::Vector &x, mfem::Vector &y) const {
		total_timer.Clear(); total_timer.Start(); 
		S.Mult(x, y); 
		D.MultTranspose(y, psi); 
		psi += source; 

		sweep_timer.Clear(); sweep_timer.Start(); 
		Linv.Mult(psi, psi); 
		sweep_timer.Stop(); 

		moment_timer.Clear(); moment_timer.Start(); 
		moment.Mult(psi, y); 
		moment_timer.Stop(); 

		total_timer.Stop(); 
	}

	auto &TotalTimer() const { return total_timer; }
	auto &SweepTimer() const { return sweep_timer; }
	auto &MomentTimer() const { return moment_timer; }
};