#pragma once 

#include "mfem.hpp"
#include "log.hpp"

// extract diagonal blocks according to element dofs and invert
// caller must delete ptr 
mfem::HypreParMatrix *ElementByElementBlockInverse(const mfem::ParFiniteElementSpace &fes, const mfem::HypreParMatrix &A); 
// combine a block operator of hypre matrices into a single hypre matrix 
// caller must delete ptr 
mfem::HypreParMatrix *BlockOperatorToMonolithic(const mfem::BlockOperator &bop); 

class ComponentExtractionOperator : public mfem::Operator 
{
private:
	mfem::Array<int> offsets; 
	int comp; 
public:
	ComponentExtractionOperator(const mfem::Array<int> &_offsets, int c) 
		: offsets(_offsets), comp(c) 
	{
		height = offsets[comp+1] - offsets[comp]; 
		width = offsets.Last(); 
	}
	void Mult(const mfem::Vector &x, mfem::Vector &y) const {
		mfem::BlockVector bx(*const_cast<mfem::Vector*>(&x), 0, offsets); 
		y = bx.GetBlock(comp);
	}
	void MultTranspose(const mfem::Vector &x, mfem::Vector &y) const {
		y = 0.0; 
		mfem::BlockVector by(y.GetData(), offsets); 
		by.GetBlock(comp) = x; 
	}
};

class SubBlockExtractionOperator : public mfem::Operator 
{
private:
	mfem::BlockVector &block_data; 
	mfem::Array<int> offsets; 
	int comp; 
public:
	SubBlockExtractionOperator(mfem::BlockVector &data, int c) 
		: block_data(data), comp(c) 
	{
		offsets.SetSize(block_data.NumBlocks()+1); 
		offsets[0] = 0; 
		for (auto i=0; i<block_data.NumBlocks(); i++) {
			offsets[i+1] = block_data.BlockSize(i); 
		}
		offsets.PartialSum(); 
		height = block_data.BlockSize(comp); 
		width = offsets.Last(); 
	}
	void Mult(const mfem::Vector &x, mfem::Vector &y) const {
		assert(x.Size() == width); 
		assert(y.Size() == height); 
		mfem::BlockVector bx(*const_cast<mfem::Vector*>(&x), offsets); 
		block_data = bx; 
		y = block_data.GetBlock(comp); 
	}
	void MultTranspose(const mfem::Vector &x, mfem::Vector &y) const {
		if (y.GetData() != block_data.GetData()) {
			mfem::BlockVector by(y, offsets);
			by = block_data;
		}
	}
};

// applies an operator to one component of a BlockVector 
// and returns the rest as is
// intended to reduce the memory cost of iterating on 
// solution vectors including the angular flux 
// such as for converting [psi,T] -> [E,T] 
// to avoid storing and computing norms on psi 
class SubBlockReductionOperator : public mfem::Operator
{
private:
	const mfem::Operator &op;
	mfem::BlockVector &data;
	mfem::Array<int> row_offsets, col_offsets;
	int comp;

	mfem::Array<const mfem::Operator*> opers;
public:
	SubBlockReductionOperator(mfem::BlockVector &data, const mfem::Operator &op, int c);
	void Mult(const mfem::Vector &x, mfem::Vector &y) const override;
	// returns the full BlockVector in y 
	// checks if y and data are aliased to each other to avoid 
	// unnecessary copy 
	void MultTranspose(const mfem::Vector &x, mfem::Vector &y) const override;
};

// reimplement mfem::TripleProductOperator but initialize temporary vectors 
// to zero so that operator A or B can be an iterative solver 
// with iterative_mode = true 
class TripleProductOperator : public mfem::Operator 
{
private:
	const mfem::Operator *A, *B, *C; 
	bool ownA, ownB, ownC; 
	mutable mfem::Vector t1, t2;
	mutable mfem::StopWatch timer; 
	bool log_time = false; 
	std::array<std::string,3> timing_keys; 
public:
	TripleProductOperator(const mfem::Operator *a, const mfem::Operator *b, const mfem::Operator *c,
		bool owna, bool ownb, bool ownc); 
	~TripleProductOperator(); 
	void Mult(const mfem::Vector &x, mfem::Vector &y) const {
		timer.Restart(); 
		C->Mult(x,t1); 
		timer.Stop(); 
		if (log_time) TimingLog[timing_keys[2]] += timer.RealTime(); 

		timer.Restart(); 		
		B->Mult(t1, t2); 
		timer.Stop(); 
		if (log_time) TimingLog[timing_keys[1]] += timer.RealTime(); 

		timer.Restart(); 
		A->Mult(t2, y); 
		timer.Stop(); 
		if (log_time) TimingLog[timing_keys[0]] += timer.RealTime(); 
	}
	void MultTranspose(const mfem::Vector &x, mfem::Vector &y) const {
		A->MultTranspose(x, t2); B->MultTranspose(t2, t1); C->MultTranspose(t1, y); 
	}

	void SetLoggingKeys(std::string a, std::string b, std::string c) {
		timing_keys[0] = a; 
		timing_keys[1] = b; 
		timing_keys[2] = c; 
		log_time = true; 
	}
};

// re-implement mfem::SLISolver since they don't call monitor 
// and don't count number of iterations correctly 
class SLISolver : public mfem::IterativeSolver {
private:
	mutable mfem::Vector r,z; 
public:
	SLISolver(MPI_Comm _comm) : mfem::IterativeSolver(_comm) { }
	SLISolver() = default;
	void SetOperator(const mfem::Operator &op) { 
		mfem::IterativeSolver::SetOperator(op); 
		r.SetSize(width); 
		z.SetSize(width); 
	}
	void Mult(const mfem::Vector &b, mfem::Vector &x) const; 
};

// forwards the source term provided in Mult to 
// the nested class FixedPointOperator 
// this makes a fixed point solver look like 
// a standard linear solver that acts on a source term 
// and returns a solution 
class FixedPointSolverWrapper : public mfem::Operator {
private:
	const mfem::Operator &op;
	mfem::Solver &fp_solver;
public:
	FixedPointSolverWrapper(const mfem::Operator &op, mfem::Solver &fp_solver)
		: op(op), fp_solver(fp_solver)
	{
		height = op.Height();
		width = op.Width();
	}
	void Mult(const mfem::Vector &b, mfem::Vector &x) const override
	{
		FixedPointOperator fp_op(op, b);
		fp_solver.SetOperator(fp_op);
		mfem::Vector blank;
		fp_solver.Mult(blank, x);
	}

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


// [A, B] = [I  B D^{-1}] [ S 0 ] [I        0 ]
// [C, D]   [0      I   ] [ 0 D ] [D^{-1} C I ]
//        = [I  B D^{-1}] [S 0]
//          [0      I   ] [C D]
//        = [S + B D^{-1} C  B ]
//          [     C          D ]
//        = [A B] 
//          [C D]
// S = A - B D^{-1} C 
class BlockLDUInverseOperator : public mfem::Operator
{
private:
	const mfem::Operator &Sinv, &Dinv, &B, &C; 
	mfem::Array<int> offsets; 
	mutable mfem::BlockVector tmp; 
public:
	BlockLDUInverseOperator(const mfem::Operator &sinv, const mfem::Operator &dinv, 
		const mfem::Operator &b, const mfem::Operator &c)
		: Sinv(sinv), Dinv(dinv), B(b), C(c)
	{
		offsets.SetSize(3); 
		offsets[0] = 0; 
		offsets[1] = Sinv.Height(); 
		offsets[2] = Dinv.Height(); 
		offsets.PartialSum(); 
		height = width = offsets.Last(); 
	}
	void Mult(const mfem::Vector &b, mfem::Vector &x) const; 
};

// wrap mfem::SuperLUSolver to modify SetOperator behavior 
// to first create a mfem::SuperLURowLocMatrix 
// enabling this SuperLUSolver to function like 
// any other sparse solver 
// NOTE: doubles storage of operator since both 
// the original operator and SuperLURowLocMatrix 
// copy 
class SuperLUSolver : public mfem::Solver 
{
private:
	mfem::SuperLUSolver slu;
	std::unique_ptr<mfem::SuperLURowLocMatrix> slu_op;
public:
	SuperLUSolver(MPI_Comm comm) : slu(comm) {
		slu.SetPrintStatistics(false);
	}
	void SetOperator(const mfem::Operator &op) override;
	void Mult(const mfem::Vector &b, mfem::Vector &x) const override {
		slu.Mult(b, x);
	}

	mfem::SuperLUSolver &GetSolver() { return slu; }
	const mfem::SuperLUSolver &GetSolver() const { return slu; }
};

class BlockDiagonalPreconditioner : public mfem::Solver
{
private:
	int nBlocks;
	mfem::Array<mfem::Solver*> solvers;
	mfem::Array<int> offsets;

	mutable mfem::BlockVector xBlock, bBlock;
public:
	BlockDiagonalPreconditioner(int nBlocks);
	~BlockDiagonalPreconditioner();
	void SetOperator(const mfem::Operator &op) override; 
	void Mult(const mfem::Vector &b, mfem::Vector &x) const override;
	void SetDiagonalBlock(int iBlock, mfem::Solver &solver);
	mfem::Solver &GetDiagonalBlock(int iBlock) { return *solvers[iBlock]; }
	const mfem::Solver &GetDiagonalBlock(int iBlock) const { return *solvers[iBlock]; }

	int owns_blocks = 0;
};

class PARSolver : public mfem::Solver
{
private:
	const mfem::Operator &R, &P; 
	mfem::Solver &A;

	mutable mfem::Vector b_restricted, x_restricted;
public:
	PARSolver(const mfem::Operator &R, mfem::Solver &A, const mfem::Operator &P)
		: R(R), P(P), A(A)
	{
		height = P.Width();
		width = R.Width();

		b_restricted.SetSize(R.Height());
		x_restricted.SetSize(R.Height());
		x_restricted = 0.0;
	}
	void SetOperator(const mfem::Operator &op)
	{
		A.SetOperator(op);
		assert(R.Height() == A.Width());
		assert(P.Height() == A.Height());
	}
	void Mult(const mfem::Vector &b, mfem::Vector &x) const
	{
		R.Mult(b, b_restricted);
		if (A.iterative_mode)
			R.Mult(x, x_restricted);
		A.Mult(b_restricted, x_restricted);
		P.MultTranspose(x_restricted, x);
	}
};