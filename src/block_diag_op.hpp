#pragma once 

#include "mfem.hpp"

// stores diagonal blocks in an array to avoid CSR indexing 
// Mult loops through elements and applies the block 
// associated with that element 
class DenseBlockDiagonalOperator : public mfem::Operator
{
protected:
	const mfem::Table *row_table, *col_table;
	mfem::Array<mfem::DenseMatrix*> data;
public:
	DenseBlockDiagonalOperator(const mfem::Table &row_table, const mfem::Table &col_table);
	DenseBlockDiagonalOperator(const mfem::FiniteElementSpace &fes);
	DenseBlockDiagonalOperator(const mfem::FiniteElementSpace &tr_fes, const mfem::FiniteElementSpace &te_fes);
	int NumBlocks() const { return data.Size(); }
	const mfem::Table &GetRowDofs() const { return *row_table; }
	const mfem::Table &GetColDofs() const { return *col_table; }
	const mfem::DenseMatrix &GetBlock(int block) const;
	mfem::DenseMatrix &GetBlock(int block);
	void SetBlock(int block, const mfem::DenseMatrix &elmat);
	void Mult(const mfem::Vector &x, mfem::Vector &y) const override;
	void AddMult(const mfem::Vector &x, mfem::Vector &y, const double a=1.0) const override;

	void Invert();
	mfem::SparseMatrix *AsSparseMatrix() const;
};

void add(double a, const DenseBlockDiagonalOperator &A, 
	double b, const DenseBlockDiagonalOperator &B, DenseBlockDiagonalOperator &C);
void Mult(const DenseBlockDiagonalOperator &A, const DenseBlockDiagonalOperator &B, 
	DenseBlockDiagonalOperator &C);
void Mult(const DenseBlockDiagonalOperator &A, const mfem::SparseMatrix &B, 
	DenseBlockDiagonalOperator &C);
mfem::SparseMatrix *Mult(const DenseBlockDiagonalOperator &A, const mfem::SparseMatrix &B);
mfem::SparseMatrix *TripleProduct(const DenseBlockDiagonalOperator &A, const DenseBlockDiagonalOperator &B, 
	const mfem::SparseMatrix &C);
mfem::SparseMatrix *RAP(const DenseBlockDiagonalOperator &A, const mfem::SparseMatrix &P);

// apply a linear solver to each element 
// local solver is intended to be one of mfem::DenseMatrixInverse 
// or DiagonalDenseMatrixInverse
class DenseBlockDiagonalSolver : public mfem::Solver
{
private:
	mfem::Solver &local_solver; 
	const DenseBlockDiagonalOperator *op = nullptr; 
public:
	DenseBlockDiagonalSolver(mfem::Solver &ls) 
		: local_solver(ls), mfem::Solver(0, false)
	{ }
	void SetOperator(const mfem::Operator &op) override; 
	void Mult(const mfem::Vector &b, mfem::Vector &x) const override; 
};

// inverts mfem::DenseMatrix under the assumption that 
// the matrix is diagonal 
// used to invert lumped matrices efficiently 
// ensures matrix is diagonal in debug mode 
class DiagonalDenseMatrixInverse : public mfem::Solver
{
private:
	const mfem::DenseMatrix *mat = nullptr; 
public:
	DiagonalDenseMatrixInverse() = default; 
	void SetOperator(const mfem::Operator &op) override; 
	void Mult(const mfem::Vector &x, mfem::Vector &y) const override; 
};

class DenseBlockDiagonalNonlinearForm : public mfem::NonlinearForm
{
private:
	mutable DenseBlockDiagonalOperator *grad = nullptr;
public:
	DenseBlockDiagonalNonlinearForm(mfem::FiniteElementSpace *f)
		: mfem::NonlinearForm(f)
	{ }
	~DenseBlockDiagonalNonlinearForm(); 
	// reimplement to fix bug in mfem 
	void Mult(const mfem::Vector &x, mfem::Vector &y) const override; 
	// reimplement to return block diagonal operator 
	DenseBlockDiagonalOperator &GetGradient(const mfem::Vector &x) const override;

	// restrict the global nonlinear form to a single element 
	// facilitates element-local solves 
	class LocalOperator : public mfem::Operator 
	{
	private:
		const DenseBlockDiagonalNonlinearForm &form; 
		int element;
		mutable mfem::DenseMatrix grad; 
	public:
		LocalOperator(const DenseBlockDiagonalNonlinearForm &form, int e) 
			: form(form), element(e)
		{
			height = width = form.fes->GetFE(element)->GetDof() * form.fes->GetVDim(); 
		}
		void Mult(const mfem::Vector &x, mfem::Vector &y) const override 
		{
			form.AssembleLocalResidual(element, x, y); 
		}
		mfem::Operator &GetGradient(const mfem::Vector &x) const override
		{
			form.AssembleLocalGradient(element, x, grad); 
			return grad; 
		}
	};
	void AssembleLocalResidual(int e, const mfem::Vector &x, mfem::Vector &y) const; 
	void AssembleLocalGradient(int e, const mfem::Vector &x, mfem::DenseMatrix &A) const;
};

// newton solver designed to solve the element-local energy balance
// implements under relaxation to reduce negativities and
// checks for solution stagnation to prevent unnecessary iterations
class EnergyBalanceNewtonSolver : public mfem::NewtonSolver
{
public:
	// track why iteration stopped 
	enum ExitCode {
		Undefined = -10,
		MaxIterationsReached = 0,
		Success = 1, 
		Stagnated = 2
	};
private:
	// under relax if solution dips below minimum_solution 
	double minimum_solution = 1e-8; 
	// amount to under relax by 
	double under_relax_param = 0.1; 
	// break if solution doesn't change max_stagnation_count
	// iterations in a row 
	static constexpr int max_stagnation_count = 3;
	// store new solution while checking for positivity 
	mutable mfem::Vector xnew, successive_residual; 
	// track exit condition for last call to Mult 
	mutable ExitCode exit_code = Undefined;
public:
	EnergyBalanceNewtonSolver() = default; 
	void SetOperator(const mfem::Operator &op) override; 
	void Mult(const mfem::Vector &b, mfem::Vector &x) const override; 

	ExitCode GetExitCode() const { return exit_code; } 
	void SetMinimumSolution(double m) { minimum_solution = m; }
	void SetUnderRelaxParameter(double v) { under_relax_param = v; }
};

// calls a nonlinear solver on each element 
// collects convergence information from local solve 
// reported statistics are the maximum values achieved across 
// all elements 
class DenseBlockDiagonalNonlinearSolver : public mfem::Solver
{
private:
	mfem::IterativeSolver &local_solver; 
	const DenseBlockDiagonalNonlinearForm *form = nullptr; 

	// conform to mfem::IterativeSolver interface 
	mutable bool converged = false; 
	mutable int final_iter = -1;
	mutable double final_rel_norm = -1.0, final_norm = -1.0; 
public:
	DenseBlockDiagonalNonlinearSolver(mfem::IterativeSolver &ls)
		: local_solver(ls)
	{ }
	void SetOperator(const mfem::Operator &op) override;
	void Mult(const mfem::Vector &b, mfem::Vector &x) const override; 

	bool GetConverged() const { return converged; }
	int GetNumIterations() const { return final_iter; }
	double GetFinalRelNorm() const { return final_rel_norm; }
	double GetFinalNorm() const { return final_norm; }
};