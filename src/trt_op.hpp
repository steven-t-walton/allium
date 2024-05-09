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

class PicardTRTOperator : public mfem::TimeDependentOperator
{
private:
	const mfem::Array<int> &offsets; 
	const mfem::Operator &Linv, &B, &C;
	const mfem::Solver &meb_solver;
	mfem::Solver &schur_solver;
	mutable mfem::Vector t1, t2; 
public:
	PicardTRTOperator(
		const mfem::Array<int> &offsets, 
		const mfem::Operator &Linv, 
		const mfem::Operator &B, 
		const mfem::Operator &C, 
		const mfem::Solver &meb_solver,
		mfem::Solver &schur_solver);
	void Mult(const mfem::Vector &b, mfem::Vector &x) const override; 

private:
	class SchurComplementOperator : public mfem::Operator
	{
	private:
		const mfem::Operator &Linv, &B, &C; 
		const mfem::Solver &meb_solver; 
		const mfem::Vector &source;
		mfem::Vector &tmp, &psi;  
	public:
		SchurComplementOperator(
			const mfem::Operator &Linv, 
			const mfem::Operator &B, 
			const mfem::Operator &C, 
			const mfem::Solver &meb_solver,
			const mfem::Vector &source, 
			mfem::Vector &tmp, 
			mfem::Vector &psi
			)
			: Linv(Linv), B(B), C(C), meb_solver(meb_solver), source(source), tmp(tmp), psi(psi)
		{
			height = width = meb_solver.Height(); 
		}
		void Mult(const mfem::Vector &x, mfem::Vector &y) const override; 
	};
};