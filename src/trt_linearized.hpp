#pragma once 

#include "mfem.hpp" 
#include "sweep.hpp"

// solves transport and energy balance with Newton 
// psi is linearly eliminated to avoid computing 
// and storing a psi-sized residual in the Newton solve 
// [  L  -B         ]   [ I             ] [L      -B            ] = [ source_psi ]
// [     Bt  -sigma ] = [          I    ] [       Bt     -sigma ] = [  source_T  ]
// [ -D         I   ]   [ -D Linv     I ] [   -D Linv B     I   ] = [      0     ]
// the reduced nonlinear system 
// [     Bt    -sigma ]
// [ -D Linv B    I   ]
// is solved with Newton iteration 
// the Jacobian is 
// [     dBt     -sigma ] = [         I              ] [ dBt                -sigma         ]
// [ -D Linv dB     I   ]   [ -D Linv dB dBt^{-1}  I ] [      I - D Linv dB dBt^{-1} sigma ]
// the schur complement looks like a steady-state transport problem with 
// pseudo scattering dB dBt^{-1} sigma => solve with linear solver (eg gmres) and
// an optional DSA preconditioner 
class NewtonTRTOperator : public mfem::Operator
{
private:
	const mfem::Array<int> &offsets; 
	const InverseAdvectionOperator &Linv; 
	const mfem::Operator &D, &B, &Bt, &sigma;
	mfem::IterativeSolver &nonlinear_solver, &schur_solver; 
	mfem::Solver &meb_grad_inv; 
	const mfem::Solver *dsa_solver = nullptr; 

	mfem::Array<int> reduced_offsets; // size of [ phi, T ] for Newton operator 
	mutable std::unique_ptr<mfem::BlockVector> reduced_x, reduced_b; 
	mfem::IterativeSolver *rebalance_solver = nullptr; 
public:
	NewtonTRTOperator(
		const mfem::Array<int> &offsets, // [psi, T]
		const InverseAdvectionOperator &Linv, // sweep
		const mfem::Operator &D, // discrete to moment 
		const mfem::Operator &B, // emission
		const mfem::Operator &Bt, // emission with Cv/dt term
		const mfem::Operator &sigma, // M_sigma 
		mfem::IterativeSolver &nonlinear_solver, // solves reduces [T, phi] system 
		mfem::IterativeSolver &schur_solver, // solves linear transport problem 
		mfem::Solver &meb_grad_inv, // inverts Bt 
		const mfem::Solver *dsa_solver=nullptr // inverts diffusion system for DSA 
		); 
	void Mult(const mfem::Vector &x, mfem::Vector &y) const override; 

	// if set, solves local temperature equation in case Newton did not converge 
	// or a fix up is used 
	void SetRebalanceSolver(mfem::IterativeSolver &op) { rebalance_solver = &op; }

	// block operator that applies 
	// [    I    -D Linv B ]
	// [ -sigma      Bt    ]
	// produces gradient operator 
	// [    I    -D Linv dB ]
	// [ -sigma      dBt    ]
	class NonlinearOperator : public mfem::Operator {
	private:
		const mfem::Array<int> &offsets; 
		const mfem::Operator &Linv, &D, &B, &Bt, &sigma; 
		mfem::Vector &psi; 

		mutable mfem::Vector tmp;
		mutable std::unique_ptr<mfem::BlockOperator> grad; 
		mutable std::unique_ptr<mfem::Operator> F11, F12; 
	public:
		NonlinearOperator(
			const mfem::Array<int> &offsets, 
			const mfem::Operator &Linv, 
			const mfem::Operator &D, 
			const mfem::Operator &B, 
			const mfem::Operator &Bt, 
			const mfem::Operator &sigma, 
			mfem::Vector &psi);
		void Mult(const mfem::Vector &x, mfem::Vector &y) const override; 
		mfem::Operator &GetGradient(const mfem::Vector &x) const override; 
	}; 

	// solves NonlinearOperator::GetGradient with block LU 
	// [ I  -D Linv dB (dBt)^-1 ] [ I - D Linv dB (dBt)^-1 sigma        ]
	// [                I       ] [          -sigma                dBt  ]
	class JacobianSolver : public mfem::Solver {
	private:
		const mfem::Array<int> &offsets; 
		mfem::IterativeSolver &schur_solver;
		mfem::Solver &meb_grad_inv; 
		const mfem::Solver *dsa_solver = nullptr;

		const mfem::BlockOperator *block_op = nullptr; 
		std::unique_ptr<mfem::Operator> schur_op, dsa_Ms; 
		std::unique_ptr<mfem::Solver> dsa_op; 
		mutable mfem::Vector tmp_phi, tmp_T; 
	public:
		JacobianSolver(const mfem::Array<int> &offsets, 
			mfem::IterativeSolver &schur_solver, mfem::Solver &meb_grad_inv, 
			const mfem::Solver *dsa_solver=nullptr);
		void SetOperator(const mfem::Operator &op) override; 
		void Mult(const mfem::Vector &b, mfem::Vector &x) const override; 
	};
private:
	// applies D Linv D^T dB 
	// uses working vector to avoid psi-sized allocation
	class F12Operator : public mfem::Operator {
	private:
		const mfem::Operator &D, &Linv, &dB; 
		mfem::Vector &psi; 
	public:
		F12Operator(
			const mfem::Operator &D, 
			const mfem::Operator &Linv, 
			const mfem::Operator &dB, 
			mfem::Vector &psi)
			: D(D), Linv(Linv), dB(dB), psi(psi) 
		{
			height = D.Height(); 
			width = dB.Width(); 
		}
		void Mult(const mfem::Vector &x, mfem::Vector &y) const override {
			dB.Mult(x, y); 
			D.MultTranspose(y, psi);
			Linv.Mult(psi, psi); 
			D.Mult(psi, y); 
		}
		// DSA needs access to dB to form pseudo scattering operator 
		const mfem::Operator &GetEmissionGradient() const { return dB; }
	};
	// applies I - D Linv dB dBt^{-1} sigma 
	class SchurComplementOperator : public mfem::Operator {
	private:
		const mfem::Operator &A, &B, &C; 
		mutable mfem::Vector t1, t2; 
	public:
		SchurComplementOperator(const mfem::Operator &A, 
			const mfem::Operator &B, const mfem::Operator &C)
			: A(A), B(B), C(C)
		{
			assert(C.Height() == B.Width()); 
			assert(B.Height() == A.Width()); 
			height = A.Height(); 
			width = C.Width(); 
			t1.SetSize(C.Height()); 
			t2.SetSize(B.Height()); 
		}
		void Mult(const mfem::Vector &x, mfem::Vector &y) const override
		{
			C.Mult(x, t1); 
			B.Mult(t1, t2); 
			A.Mult(t2, y); 
			add(x, -1.0, y, y); // x - y -> y 
		}
	};
};

class LinearizedTRTOperator : public mfem::Operator
{
private:
	const mfem::Array<int> &offsets; 
	const InverseAdvectionOperator &Linv; 
	const mfem::Operator &D, &B, &Bt, &sigma;
	mfem::IterativeSolver &schur_solver; 
	mfem::Solver &meb_grad_inv; 
	const mfem::Solver *dsa_solver = nullptr; 

	mutable mfem::Vector temp_resid, dT, em_source, phi_source, phi; 
	mfem::IterativeSolver *rebalance_solver = nullptr; 
public:
	LinearizedTRTOperator(
		const mfem::Array<int> &offsets, // [psi, T]
		const InverseAdvectionOperator &Linv, // sweep
		const mfem::Operator &D, // discrete to moment 
		const mfem::Operator &B, // emission
		const mfem::Operator &Bt, // emission with Cv/dt term
		const mfem::Operator &sigma, // M_sigma 
		mfem::IterativeSolver &schur_solver, // solves linear transport problem 
		mfem::Solver &meb_grad_inv, // inverts Bt 
		const mfem::Solver *dsa_solver=nullptr // inverts diffusion system for DSA 
		); 
	void Mult(const mfem::Vector &x, mfem::Vector &y) const override; 

	// if set, solves local temperature equation in case Newton did not converge 
	// or a fix up is used 
	void SetRebalanceSolver(mfem::IterativeSolver &op) { rebalance_solver = &op; }
};