#pragma once 

#include "mfem.hpp"
#include "tvector.hpp"
#include "opacity.hpp"
#include "sweep.hpp"

class MultiGroupBilinearForm;

// solves block system
// [ L  -X  ] = [    I        ] [ L          -X          ]
// [ -Y  Bt ]   [ -Y Linv   I ] [         Bt - Y Linv X  ]
// where X = D^T emission, Y = M_sigma D
// schur complement Bt - Y Linv X is inverted with 
// an iterative fixed point solver 
class PicardTRTOperator : public mfem::Operator
{
private:
	const mfem::Array<int> &offsets; 
	InverseAdvectionOperator &Linv; 
	const mfem::Solver &meb_solver;
	mfem::Solver &schur_solver;

	mfem::TransposeOperator DT; 
	mfem::ProductOperator B, C; 
	mutable mfem::Vector t1, t2; 

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
private:
	// applies T^{k+1} = Bt^{-1}(source + Y Linv X(T^k))
	class FixedPointOperator : public mfem::Operator
	{
	private:
		InverseAdvectionOperator &Linv;
		const mfem::Operator &B, &C; 
		const mfem::Solver &meb_solver; 
		const mfem::Vector &source_psi, &source_T;
		mfem::Vector &tmp, &psi;  

		ProjectedVectorCoefficient *opacity = nullptr;
		MultiGroupBilinearForm *Mtot = nullptr;
	public:
		FixedPointOperator(
			InverseAdvectionOperator &Linv, 
			const mfem::Operator &B, 
			const mfem::Operator &C, 
			const mfem::Solver &meb_solver,
			const mfem::Vector &source_psi, 
			const mfem::Vector &source_T,
			mfem::Vector &tmp, // temporary vector size of temperature
			mfem::Vector &psi // temporary vector size of psi 
			)
			: Linv(Linv), B(B), C(C), meb_solver(meb_solver), source_psi(source_psi), source_T(source_T), tmp(tmp), psi(psi)
		{
			height = width = meb_solver.Height(); 
		}
		void UseImplicitOpacity(ProjectedVectorCoefficient &opac, MultiGroupBilinearForm &M) 
		{
			opacity = &opac;
			Mtot = &M;
		}
		void Mult(const mfem::Vector &x, mfem::Vector &y) const override; 
	};
};