#include "fixed_point.hpp"

ComponentReductionOperator::ComponentReductionOperator(const mfem::Array<int> &offsets, int c)
	: SolutionReductionOperator(offsets), comp(c)
{
	height = offsets[comp+1] - offsets[comp];
	oper = new mfem::IdentityOperator(height);
	own_oper = true;
}

ComponentReductionOperator::ComponentReductionOperator(const mfem::Array<int> &offsets, const mfem::Operator &op, int c)
	: SolutionReductionOperator(offsets), oper(&op), comp(c)
{
	height = oper->Height();
	own_oper = false;
	assert(offsets[comp+1] - offsets[comp] == oper->Width());
}

ComponentReductionOperator::~ComponentReductionOperator()
{
	if (own_oper) delete oper;
}

void ComponentReductionOperator::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
	assert(data);
	assert(y.Size() == oper->Height());
	*data = x;
	const mfem::BlockVector bx(const_cast<mfem::Vector&>(x), offsets);
	oper->Mult(bx.GetBlock(comp), y);
}

void ComponentReductionOperator::MultTranspose(const mfem::Vector &x, mfem::Vector &y) const
{
	assert(data);
	mfem::BlockVector by(y, offsets);
	oper->MultTranspose(x, by.GetBlock(comp));
}

ProjectGridFunctionOperator::ProjectGridFunctionOperator(mfem::FiniteElementSpace &in_fes, mfem::FiniteElementSpace &out_fes)
	: in_fes(in_fes), out_fes(out_fes)
{
	height = out_fes.GetVSize();
	width = in_fes.GetVSize();
}

void ProjectGridFunctionOperator::Mult(const mfem::Vector &x, mfem::Vector &y) const 
{
	const mfem::GridFunction gfx(&in_fes, const_cast<mfem::Vector&>(x), 0);
	mfem::GridFunctionCoefficient gfx_coef(&gfx);
	mfem::GridFunction gfy(&out_fes, y, 0);
	gfy.ProjectCoefficient(gfx_coef);
}

void ProjectGridFunctionOperator::MultTranspose(const mfem::Vector &x, mfem::Vector &y) const 
{
	const mfem::GridFunction gfx(&out_fes, const_cast<mfem::Vector&>(x), 0);
	mfem::GridFunctionCoefficient gfx_coef(&gfx);
	mfem::GridFunction gfy(&in_fes, y, 0);
	gfy.ProjectCoefficient(gfx_coef);
}

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
	auto *kinsol = dynamic_cast<mfem::KINSolver*>(&fp_solver);
	if (kinsol) {
		mfem::Vector fx_scale(height), x_scale(height);
		oper->Mult(b, x);
		double norm = x.Normlinf();
		double gnorm;
		MPI_Allreduce(&norm, &gnorm, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		fx_scale = 1.0 / gnorm;
		x_scale = 1.0;
		kinsol->Mult(x, x_scale, fx_scale);
	} else {
		mfem::Vector blank;
		fp_solver.Mult(blank, x);
	}
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
	reducer.MultTranspose(xred, xfull);
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
	double initial_mag;
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
			initial_mag = Norm(x);
			r0 = norm *rel_tol;
		}

		if (norm < r0) {
			converged = true; 
			final_iter = i; 
		}

		else if (norm / initial_mag < abs_tol) {
			converged = true; 
			final_iter = i;
			norm /= initial_mag;
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