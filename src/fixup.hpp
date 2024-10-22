#pragma once 

#include "mfem.hpp"

class NegativeFluxFixupOperator 
{
protected:
	double minimum_solution = 0.0; 
public:
	NegativeFluxFixupOperator() = default; 
	NegativeFluxFixupOperator(double min) : minimum_solution(min) { }
	virtual bool Apply(const mfem::DenseMatrix &A, const mfem::Vector &rhs, mfem::Vector &solution) const =0;
};

class ZeroFixupOperator : public NegativeFluxFixupOperator
{
public:
	ZeroFixupOperator(double min) : NegativeFluxFixupOperator(min) { }
	bool Apply(const mfem::DenseMatrix &A, const mfem::Vector &rhs, mfem::Vector &solution) const override;	
};

class ZeroAndScaleFixupOperator : public NegativeFluxFixupOperator
{
private:
	mutable mfem::Vector weights; 
public:
	ZeroAndScaleFixupOperator(double min) : NegativeFluxFixupOperator(min) { }
	bool Apply(const mfem::DenseMatrix &A, const mfem::Vector &rhs, mfem::Vector &solution) const override;
};

class LocalOptimizationFixupOperator : public NegativeFluxFixupOperator
{
private:
	mfem::SLBQPOptimizer &opt; 
	mutable mfem::Vector ones, weights, low, high, source; 
public:
	LocalOptimizationFixupOperator(mfem::SLBQPOptimizer &opt, double min=0.0) 
		: opt(opt), NegativeFluxFixupOperator(min)
	{ }
	bool Apply(const mfem::DenseMatrix &A, const mfem::Vector &rhs, mfem::Vector &solution) const override;
};

class RyosukeFixupOperator : public NegativeFluxFixupOperator
{
public:
	RyosukeFixupOperator(double min) : NegativeFluxFixupOperator(min) { }
	bool Apply(const mfem::DenseMatrix &A, const mfem::Vector &rhs, mfem::Vector &solution) const override;
private:
	unsigned perform_nff(unsigned sz, double *x, double *A, double *b) const;
};