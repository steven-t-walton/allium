#pragma once 

#include "mfem.hpp"

class BlockDiagonalByElementOperator : public mfem::Operator
{
private:
	const mfem::FiniteElementSpace &fes; 
	mfem::Array<mfem::DenseMatrix*> data; 
public:
	BlockDiagonalByElementOperator(const mfem::FiniteElementSpace &f);
	~BlockDiagonalByElementOperator(); 
	const mfem::FiniteElementSpace *FESpace() const { return &fes; }
	void SetElementMatrix(int elem, const mfem::DenseMatrix &elmat); 
	const mfem::DenseMatrix &GetElementMatrix(int elem) const; 
	mfem::DenseMatrix &GetElementMatrix(int elem); 

	void Mult(const mfem::Vector &x, mfem::Vector &y) const override; 
};

class BlockDiagonalByElementSolver : public mfem::Solver
{
private:
	bool assume_diagonal;
	const BlockDiagonalByElementOperator *op = nullptr; 
public:
	BlockDiagonalByElementSolver(bool assume_diagonal=false)
		: assume_diagonal(assume_diagonal), mfem::Solver(0, false)
	{ }
	void SetOperator(const mfem::Operator &op) override; 
	void Mult(const mfem::Vector &x, mfem::Vector &y) const override; 
};

class BlockDiagonalByElementNonlinearForm : public mfem::NonlinearForm
{
private:
	mutable BlockDiagonalByElementOperator *grad = nullptr;
public:
	BlockDiagonalByElementNonlinearForm(mfem::FiniteElementSpace *f)
		: mfem::NonlinearForm(f)
	{ }
	~BlockDiagonalByElementNonlinearForm(); 
	// reimplement to fix bug in mfem 
	void Mult(const mfem::Vector &x, mfem::Vector &y) const override; 
	// reimplement to return block diagonal operator 
	mfem::Operator &GetGradient(const mfem::Vector &x) const override;
};