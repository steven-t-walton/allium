#pragma once 

#include "mfem.hpp"
#include "tvector.hpp"

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
	const mfem::Operator &Linv; 
	const mfem::Solver &meb_solver;
	mfem::Solver &schur_solver;

	mfem::TransposeOperator DT; 
	mfem::ProductOperator B, C; 
	mutable mfem::Vector t1, t2; 
public:
	PicardTRTOperator(
		const mfem::Array<int> &offsets, // size of [psi, T]
		const mfem::Operator &Linv, // sweep
		const mfem::Operator &D, // discrete to moment 
		const mfem::Operator &emission, // B(T) 
		const mfem::Operator &sigma, // M_sigma 
		const mfem::Solver &meb_solver, // nonlinearly inverts Bt 
		mfem::Solver &schur_solver // solves Bt - Y Linv X 
		);
	void Mult(const mfem::Vector &b, mfem::Vector &x) const override; 

private:
	// applies T^{k+1} = Bt^{-1}(source + Y Linv X(T^k))
	class FixedPointOperator : public mfem::Operator
	{
	private:
		const mfem::Operator &Linv, &B, &C; 
		const mfem::Solver &meb_solver; 
		const mfem::Vector &source;
		mfem::Vector &tmp, &psi;  
	public:
		FixedPointOperator(
			const mfem::Operator &Linv, 
			const mfem::Operator &B, 
			const mfem::Operator &C, 
			const mfem::Solver &meb_solver,
			const mfem::Vector &source, 
			mfem::Vector &tmp, // temporary vector size of temperature
			mfem::Vector &psi // temporary vector size of psi 
			)
			: Linv(Linv), B(B), C(C), meb_solver(meb_solver), source(source), tmp(tmp), psi(psi)
		{
			height = width = meb_solver.Height(); 
		}
		void Mult(const mfem::Vector &x, mfem::Vector &y) const override; 
	};
};