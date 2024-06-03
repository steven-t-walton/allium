#pragma once 

#include "mfem.hpp"
#include "log.hpp"

mfem::HypreParMatrix *ElementByElementBlockInverse(const mfem::ParFiniteElementSpace &fes, const mfem::HypreParMatrix &A); 

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

class FixedPointIterationSolver : public mfem::IterativeSolver {
private:
	mutable mfem::Vector xold, r, z; 
public:
	FixedPointIterationSolver(MPI_Comm _comm) : mfem::IterativeSolver(_comm) { }
	void Mult(const mfem::Vector &b, mfem::Vector &x) const; 
	void SetOperator(const mfem::Operator &op) {
		mfem::IterativeSolver::SetOperator(op); 
		xold.SetSize(width); 
		r.SetSize(width); 
	}
	void SetPreconditioner(mfem::Solver &s) {
		mfem::IterativeSolver::SetPreconditioner(s); 
		z.SetSize(width); 
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

// SuperLUSolver but SetOperator constructs 
// and stores the mfem::SuperLURowLocMatrix 
// copy of the operator 
class SuperLUSolverWrapper : public mfem::Solver
{
private:
	mfem::SuperLUSolver &slu;
	std::unique_ptr<mfem::SuperLURowLocMatrix> slu_op;
public:
	SuperLUSolverWrapper(mfem::SuperLUSolver &slu) : slu(slu)
	{
	}
	void SetOperator(const mfem::Operator &op) override {
		slu_op.reset(new mfem::SuperLURowLocMatrix(op));
		slu.SetOperator(*slu_op);
	}
	void Mult(const mfem::Vector &b, mfem::Vector &x) const override {
		slu.Mult(b, x);
	}
};