#include "fixed_point.hpp"

void FixedPointSolverWrapper::SetOperator(const mfem::Operator &op)
{
	oper = &op;
	height = op.Height();
	width = op.Width();
}

void FixedPointSolverWrapper::Mult(const mfem::Vector &b, mfem::Vector &x) const
{
	FixedPointOperator fp_op(*oper, b);
	fp_solver.SetOperator(fp_op);
	mfem::Vector blank;
	fp_solver.Mult(blank, x);
}

ReducedSolver::ReducedSolver(mfem::Solver &solver, SolutionReductionOperator &R)
	: solver(solver), reducer(R)
{
	height = width = R.Width();
	iterative_mode = true;
}

void ReducedSolver::SetOperator(const mfem::Operator &op)
{
	if (op.Height() != height or op.Width() != width) 
		MFEM_ABORT("bad size");
	oper = &op;
	rx.SetSize(reducer.Height());
}

void ReducedSolver::Mult(const mfem::Vector &b, mfem::Vector &x) const
{
	reducer.SetData(x);
	ReducedOperator Rop(*oper, reducer);
	solver.SetOperator(Rop);
	reducer.Mult(x, rx);
	solver.Mult(b, rx);
}

ReducedSolver::
ReducedOperator::ReducedOperator(const mfem::Operator &full_op, SolutionReductionOperator &reducer)
	: full_op(full_op), reducer(reducer)
{
	height = reducer.Height();
	width = full_op.Width();
	if (full_op.Height() != reducer.Width()) MFEM_ABORT("bad size");
}

void ReducedSolver::
ReducedOperator::Mult(const mfem::Vector &bfull, mfem::Vector &xred) const
{
	assert(bfull.Size() == full_op.Width());
	assert(xred.Size() == reducer.Height());
	auto &xfull = reducer.GetData();
	full_op.Mult(bfull, xfull);
	reducer.Mult(xfull, xred);
}

void FixedPointIterationSolver::Mult(const mfem::Vector &b, mfem::Vector &x) const 
{
	if (!iterative_mode) x = 0.0; 
	double norm, r0; 
	int i; 
	converged = false; 
	bool done = false; 
	for (i=1; true; ) {
		xold = x; 
		oper->Mult(xold, x); 
		subtract(x, xold, r); // r = x - xold 
		if (prec) {
			prec->Mult(r, z); // z = M*r
			x += z; // x = x + z 
			subtract(x, xold, r); // preconditioned residual 
		}
		norm = Norm(r); 
		if (i==1) {
			initial_norm = norm; 
			r0 = std::max(norm*rel_tol, abs_tol*Norm(x)); 
		}

		if (norm < r0) {
			converged = true; 
			final_iter = i; 
		}

		if (i >= max_iter or converged) {
			done = true; 
		}
		Monitor(i-1, norm, x, r, done); 

		if (done) { break; }
		i++; 
	}
	final_iter = i; 
	final_norm = norm; 
}