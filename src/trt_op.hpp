#pragma once 

#include "mfem.hpp"
#include "tvector.hpp"
#include "sweep.hpp"
#include "transport_op.hpp"

class SNTimeMassMatrix : public mfem::Operator
{
private:
	const mfem::FiniteElementSpace &fes; 
	const TransportVectorExtents &psi_ext; 
	mfem::Array<mfem::DenseMatrix*> mats; 
public:
	SNTimeMassMatrix(const mfem::FiniteElementSpace &fes, const TransportVectorExtents &ext, bool lump=false);
	~SNTimeMassMatrix() {
		for (auto *ptr : mats) { delete ptr; }
	}
	void Mult(const mfem::Vector &psi, mfem::Vector &Mpsi) const; 
};

class LocalEliminationTRTOperator : public mfem::Operator
{
private:
	const mfem::Operator &Linv, &D, &emission_form, &Mtot;
	const mfem::IterativeSolver &meb_solver;
	const mfem::Vector *source = nullptr; 
	mfem::Vector &psi;

	mutable mfem::Vector tmp; 
public:
	LocalEliminationTRTOperator(
		const mfem::Operator &Linv, 
		const mfem::Operator &D, 
		const mfem::Operator &emission_form, 
		const mfem::Operator &Mtot,
		const mfem::IterativeSolver &meb_solver, 
		mfem::Vector &psi); 
	void SetSource(const mfem::Vector &s) { source = &s; }
	void Mult(const mfem::Vector &b, mfem::Vector &x) const override; 
};

class LinearEliminationTDOperator : public mfem::TimeDependentOperator 
{
private:
	const mfem::Array<int> &offsets; 
	const mfem::Operator &Linv, &D, &emission_form, &Mtot; 
	const mfem::IterativeSolver &schur_solver; 
public:
	LinearEliminationTDOperator(
		const mfem::Array<int> &offsets,
		const mfem::Operator &Linv, 
		const mfem::Operator &D, 
		const mfem::Operator &emission_form, 
		const mfem::Operator &Mtot, 
		const mfem::IterativeSolver &schur_solver); 
	void ImplicitSolve(const double dt, const mfem::Vector &x, mfem::Vector &y) override; 
};

// class LinearizedTRTOperator : public mfem::Operator
// {
// private:
// 	const mfem::Array<int> &offsets; 
// 	const InverseAdvectionOperator &Linv; 
// 	const DiscreteToMoment &D; 
// 	const mfem::NonlinearForm &meb_form, &emission_form; 
// 	const mfem::Operator &Mtot; 
// 	mfem::IterativeSolver &schur_solver, *dsa_solver;
// public:
// 	LinearizedTRTOperator(
// 		const mfem::Array<int> &offsets, // assumes [psi, T] ordering 
// 		const InverseAdvectionOperator &Linv, // sweeper 
// 		const DiscreteToMoment &D, // psi -> phi 
// 		const mfem::NonlinearForm &meb_form, // (Cv/dt + B(.)) T 
// 		const mfem::NonlinearForm &emission_form, // B(T) 
// 		const mfem::Operator &Mtot, // sigma mass matrix 
// 		mfem::IterativeSolver &schur_solver, // invert system for phi 
// 		mfem::IterativeSolver *dsa_solver=nullptr // optional DSA solver 
// 		); 

// 	void Mult(const mfem::Vector &x, mfem::Vector &y) const override; 
// private:
// 	void SolveJacobian(mfem::Vector &x) const; 
// };