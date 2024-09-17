#pragma once 

#include "mfem.hpp"

// abstract operator that operates on block solution vectors 
// to reduce them in size 
// this class is intended for use in transport fixed point problems
// where avoiding storing extra copies and computing norms on the angular flux
// is desired 
// a pointer to the full block vector is stored, allowing
// the full solution to be recovered 
class SolutionReductionOperator : public mfem::Operator
{
protected:
	mfem::Array<int> offsets;
	mfem::Vector *data = nullptr;
public:
	SolutionReductionOperator(const mfem::Array<int> &offsets)
		: offsets(offsets)
	{
		width = offsets.Last();
	}
	virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const =0;

	void SetData(mfem::Vector &v)
	{
		if (v.Size() != Width()) MFEM_ABORT("bad size");
		data = &v;
	}
	auto &GetData() { return *data; }
	const auto &GetData() const { return *data; }
};

// returns a single component of a block vector 
class ComponentReductionOperator : public SolutionReductionOperator
{
private:
	int comp;
public:
	ComponentReductionOperator(const mfem::Array<int> &offsets, int c)
		: SolutionReductionOperator(offsets), comp(c)
	{
		height = offsets[comp+1] - offsets[comp];
	}

	void Mult(const mfem::Vector &x, mfem::Vector &y) const override
	{
		assert(data);
		*data = x;
		const mfem::BlockVector bx(const_cast<mfem::Vector&>(x), offsets);
		y = bx.GetBlock(comp);
	}
};

// forwards the source term provided in Mult to 
// the nested class FixedPointOperator 
// this makes a fixed point solver look like 
// a standard linear solver that acts on a source term 
// and returns a solution 
class FixedPointSolverWrapper : public mfem::Solver {
private:
	mfem::Solver &fp_solver;

	const mfem::Operator *oper = nullptr;
public:
	FixedPointSolverWrapper(mfem::Solver &fp_solver)
		: fp_solver(fp_solver)
	{ }
	void SetOperator(const mfem::Operator &op) override;
	void Mult(const mfem::Vector &b, mfem::Vector &x) const override;

	// converts an "inverse operator" that acts on a source term 
	// and returns a solution to a fixed point operator 
	// that takes in the previous solution iterate 
	// and returns a new one 
	class FixedPointOperator : public mfem::Operator {
	private:
		const mfem::Operator &op; 
		const mfem::Vector &source;
	public:
		FixedPointOperator(const mfem::Operator &op, const mfem::Vector &source)
			: op(op), source(source)
		{
			height = op.Height();
			width = op.Width();
			if (source.Size() != op.Width()) MFEM_ABORT("size mismatch");
		}
		void Mult(const mfem::Vector &x, mfem::Vector &y) const override
		{
			y = x;
			op.Mult(source, y);
		}
	};
};

// solver that reduces the output of 
// the operator set in SetOperator() 
// using a SolutionReductionOperator to 
// reduce memory/computation cost associated 
// with large block solution vectors 
// intended for use in fixed point iterations 
// Mult acts on the full solution vector 
// but intermediate solver iterations are 
// performed on the reduced vector 
// NOTE: does not change the solution, 
// only the stopping criterion 
class ReducedSolver : public mfem::Solver
{
private:
	mfem::Solver &solver;
	SolutionReductionOperator &reducer;

	const mfem::Operator *oper = nullptr;
	mutable mfem::Vector rx;
public:
	ReducedSolver(mfem::Solver &solver, SolutionReductionOperator &R);

	void SetOperator(const mfem::Operator &op) override;
	void Mult(const mfem::Vector &b, mfem::Vector &x) const override;
private:
	class ReducedOperator : public mfem::Operator
	{
	private:
		const mfem::Operator &full_op;
		SolutionReductionOperator &reducer;
	public:
		ReducedOperator(const mfem::Operator &full_op, SolutionReductionOperator &reducer);
		void Mult(const mfem::Vector &bfull, mfem::Vector &xred) const override;
	};
};

// solves x = G(x) using fixed point iteration 
// x^{k+1} = G(x^k) 
// operator is assumed to take in previous iterate
// and return the new iterate in the Mult call
// i.e. G = G(x^k, x^{k+1}) 
class FixedPointIterationSolver : public mfem::IterativeSolver {
private:
	mutable mfem::Vector xold, r, z; 
public:
	FixedPointIterationSolver(MPI_Comm _comm) : mfem::IterativeSolver(_comm) { }
	void Mult(const mfem::Vector &b, mfem::Vector &x) const; 
	void SetOperator(const mfem::Operator &op) {
		mfem::IterativeSolver::SetOperator(op); 
		xold.SetSize(height); 
		r.SetSize(height); 
	}
	void SetPreconditioner(mfem::Solver &s) {
		mfem::IterativeSolver::SetPreconditioner(s); 
		z.SetSize(height); 
	}
};