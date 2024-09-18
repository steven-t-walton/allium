#pragma once 

#include "mfem.hpp"
#include "tvector.hpp"
#include "opacity.hpp"
#include "sweep.hpp"

class MultiGroupBilinearForm;

class PicardTRTOperator : public mfem::Operator
{
private:
	const mfem::Array<int> &offsets; 
	InverseAdvectionOperator &Linv; 
	const mfem::Solver &meb_solver;
	mfem::Solver &schur_solver;

	mfem::TransposeOperator DT; 
	mfem::ProductOperator B, C; 
	mutable mfem::Vector tmp; 

	ProjectedVectorCoefficient *opacity = nullptr;
	MultiGroupBilinearForm *Mtot = nullptr;
public:
	PicardTRTOperator(
		const mfem::Array<int> &offsets, // size of [psi, T]
		InverseAdvectionOperator &Linv, // sweep
		const mfem::Operator &D, // discrete to moment 
		const mfem::Operator &emission, // B(T) 
		const mfem::Operator &sigma, // M_sigma 
		const mfem::Solver &meb_solver, // nonlinearly inverts Bt 
		mfem::Solver &schur_solver // solves Bt - Y Linv X 
		);
	void Mult(const mfem::Vector &b, mfem::Vector &x) const override; 

	void UseImplicitOpacity(ProjectedVectorCoefficient &opac, MultiGroupBilinearForm &M)
	{
		opacity = &opac;
		Mtot = &M;
	}	
};