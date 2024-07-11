#pragma once 
#include "mfem.hpp"
#include "multigroup.hpp"

class MultiGroupBilinearFormIntegrator 
{
protected:
	int G;
	const mfem::IntegrationRule *IntRule = nullptr;
public:
	MultiGroupBilinearFormIntegrator() = default;
	MultiGroupBilinearFormIntegrator(int G) : G(G) { }

	virtual void AssembleElementMatrices(const mfem::FiniteElement &fe, mfem::ElementTransformation &trans, 
		const mfem::Array2D<mfem::DenseMatrix*> &elmats)
	{
		MFEM_ABORT("not implemented");
	}

	void SetIntegrationRule(const mfem::IntegrationRule &rule) { IntRule = &rule; }
};

class MGMassIntegrator : public MultiGroupBilinearFormIntegrator
{
private:
	mfem::VectorCoefficient *vec_coef = nullptr;
	mfem::MatrixCoefficient *mat_coef = nullptr;
	mfem::Vector shape, vec_eval;
	mfem::DenseMatrix uu, mat_eval;
public:
	MGMassIntegrator(mfem::VectorCoefficient &sigma) 
		: vec_coef(&sigma), MultiGroupBilinearFormIntegrator(sigma.GetVDim()), vec_eval(sigma.GetVDim())
	{ }
	MGMassIntegrator(mfem::MatrixCoefficient &sigma)
		: mat_coef(&sigma), MultiGroupBilinearFormIntegrator(sigma.GetVDim()), mat_eval(sigma.GetVDim())
	{ }
	void AssembleElementMatrices(const mfem::FiniteElement &fe, mfem::ElementTransformation &trans, 
		const mfem::Array2D<mfem::DenseMatrix*> &elmats) override; 
};

class MultiGroupBilinearForm : public mfem::Operator
{
private:
	mfem::FiniteElementSpace &fes;
	int G;

	mfem::Array<int> offsets;
	mfem::Array<MultiGroupBilinearFormIntegrator*> domain_integs;
	mfem::Array2D<mfem::SparseMatrix*> spmats;
	mfem::Array2D<mfem::DenseMatrix*> elmats, elmats_all;
	mfem::BlockOperator *block_op = nullptr;
public:
	MultiGroupBilinearForm(mfem::FiniteElementSpace &fes, int G);
	~MultiGroupBilinearForm();
	void AddDomainIntegrator(MultiGroupBilinearFormIntegrator *mgbfi);
	void Assemble(int skip_zeros=1);
	void Finalize(int skip_zeros=1);
	void Mult(const mfem::Vector &x, mfem::Vector &y) const override 
	{
		assert(block_op);
		block_op->Mult(x, y);
	}
	const auto &SpMats() const { return spmats; }
	auto &SpMats() { return spmats; }
};