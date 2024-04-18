#pragma once 

#include "mfem.hpp"

class NegativeFluxFixupOperator : public mfem::Operator
{
protected:
	double minimum_solution = 0.0; 
	const mfem::DenseMatrix *A = nullptr; 
	const mfem::Vector *rhs = nullptr; 
public:
	NegativeFluxFixupOperator() = default; 
	NegativeFluxFixupOperator(double min) : minimum_solution(min) { }
	void SetLocalSystem(const mfem::DenseMatrix &a, const mfem::Vector &b) {
		A = &a; 
		rhs = &b; 
		height = width = a.Height(); 
	}
protected:
	bool DoFixup(const mfem::Vector &solution) const {
		return solution.Min() < minimum_solution; 
	}
};

class ZeroAndScaleFixupOperator : public NegativeFluxFixupOperator
{
private:
	mutable mfem::Vector ones, weights; 
public:
	ZeroAndScaleFixupOperator(double min) : NegativeFluxFixupOperator(min) { }
	void Mult(const mfem::Vector &solution, mfem::Vector &fixed) const; 
};

class LocalOptimizationFixupOperator : public NegativeFluxFixupOperator
{
private:
	mfem::SLBQPOptimizer &opt; 
	mutable mfem::Vector ones, weights, low, high; 
public:
	LocalOptimizationFixupOperator(mfem::SLBQPOptimizer &opt, double min=0.0) 
		: opt(opt), NegativeFluxFixupOperator(min)
	{ }
	void Mult(const mfem::Vector &solution, mfem::Vector &fixed) const; 
};