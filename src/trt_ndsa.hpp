#pragma once 

#include "mfem.hpp"
#include "block_diag_op.hpp"
#include "moment_discretization.hpp"
#include "opacity.hpp"

class NonlinearDSATRTOperator : public mfem::Operator
{
private:
	const mfem::Array<int> &offsets;
	const mfem::Operator &Linv, &D, &G, &B;
	const DenseBlockDiagonalNonlinearForm &Bgr, &Bt;
	mfem::BilinearForm &Mtot;
	const MomentDiscretization &lo_disc;
	mfem::Solver &linear_solver;
	mfem::IterativeSolver &lo_solver;
	const mfem::Solver &meb_solver;
	mfem::Vector &moments_nu, &moments_gray;
	mfem::Array<ProjectedCoefficient*> opacities;

	mfem::Array<int> lo_offsets;
	mutable mfem::Vector em_source, em_source_gr, phi, residual_phi, lo_source, lo_soln;
public:
	NonlinearDSATRTOperator(
		const mfem::Array<int> &offsets, // [psi, T]
		const mfem::Operator &Linv, // sweep
		const mfem::Operator &D, // discrete to moment 
		const mfem::Operator &G, // to gray 
		const mfem::Operator &B, // mg emission
		const DenseBlockDiagonalNonlinearForm &Bgr, // gray emission
		const DenseBlockDiagonalNonlinearForm &Bt, // emission with Cv/dt term
		mfem::BilinearForm &Mtot, // M_sigma 
		const MomentDiscretization &lo_disc,
		mfem::Solver &linear_solver, // solves linear transport problem 
		mfem::IterativeSolver &lo_solver,
		const mfem::Solver &meb_solver, 
		mfem::Vector &moments_nu, 
		mfem::Vector &moments_gray
		);
	template<typename... Args>
	void SetGrayOpacities(Args&... args) 
	{
		std::array<mfem::Coefficient*,sizeof...(args)> coefs{&args...};
		for (auto *coef : coefs) {
			auto *ptr = dynamic_cast<ProjectedCoefficient*>(coef);
			if (ptr) opacities.Append(ptr);
		}
	}
	void Mult(const mfem::Vector &x, mfem::Vector &y) const override;

	class LowOrderOperator : public mfem::Operator
	{
	private:
		const mfem::Array<int> &offsets;
		const mfem::HypreParMatrix &K;
		const DenseBlockDiagonalNonlinearForm &B, &Bt;
		const mfem::BilinearForm &Mtot;
		const mfem::Solver &meb_solver;
		mfem::Solver &lo_solver;

		mutable mfem::Vector emission, meb_eval, tmp_T, tmp_phi;
		mutable DenseBlockDiagonalOperator dB_dBt_inv;
	public:
		LowOrderOperator(
			const mfem::Array<int> &offsets, 
			const mfem::HypreParMatrix &K,
			const DenseBlockDiagonalNonlinearForm &B, 
			const DenseBlockDiagonalNonlinearForm &Bt, 
			const mfem::BilinearForm &Mtot, 
			const mfem::Solver &meb_solver, 
			mfem::Solver &lo_solver);
		void Mult(const mfem::Vector &source, mfem::Vector &x) const override;		
	};
};