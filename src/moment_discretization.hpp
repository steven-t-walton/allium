#pragma once 

#include "mfem.hpp"
#include "bdr_conditions.hpp"

class MomentDiscretization 
{
protected:
	mfem::ParFiniteElementSpace &fes; 
	const BoundaryConditionMap &bc_map; 
	int lumping; 

	mutable mfem::Array<int> marshak_bdr_attrs, reflect_bdr_attrs;
	double alpha = 0.5; 
	double time_absorption = -1.0;

	using HypreParMatrixPtr = std::unique_ptr<mfem::HypreParMatrix>;
	HypreParMatrixPtr Mtime;
public:
	MomentDiscretization(mfem::ParFiniteElementSpace &fes, const BoundaryConditionMap &bc_map, int lumping);
	virtual mfem::HypreParMatrix *GetOperator(mfem::Coefficient &total, mfem::Coefficient &absorption) const =0;

	void SetAlpha(double a) { alpha = a; }
	void SetTimeAbsorption(double sigma);
};

class InteriorPenaltyDiscretization : public MomentDiscretization 
{
private:
	double kappa = -1.0, mip_val = 0.0, sigma = -1.0;
public:
	InteriorPenaltyDiscretization(mfem::ParFiniteElementSpace &fes, 
		const BoundaryConditionMap &bc_map, int lumping);
	mfem::HypreParMatrix *GetOperator(mfem::Coefficient &total, mfem::Coefficient &absorption) const override;

	void SetKappa(double k) {
		if (k < 0.0) {
			const int p = fes.GetMaxElementOrder();
			kappa = -1.0 * k * pow(p+1,2);
		} else {
			kappa = k;
		}
	}
	void SetPenaltyLowerBound(double lb) { mip_val = lb; }
	void SetSigma(double s) { sigma = s; }
};

class LDGDiscretization : public MomentDiscretization
{
private:
	std::unique_ptr<mfem::ParFiniteElementSpace> vfes;
	mfem::Vector beta; 
public:
	LDGDiscretization(mfem::ParFiniteElementSpace &fes, const BoundaryConditionMap &bc_map, int lumping);
	void SetBeta(const mfem::Vector &b) { beta = b; } 
	mfem::HypreParMatrix *GetOperator(mfem::Coefficient &total, mfem::Coefficient &absorption) const override;
};

class BlockMomentDiscretization {
protected:
	mfem::ParFiniteElementSpace &fes, &vfes; 
	const BoundaryConditionMap &bc_map; 
	int lumping; 

	mfem::Array<int> offsets;
	mutable mfem::Array<int> marshak_bdr_attrs, reflect_bdr_attrs;
	double alpha = 0.5; 
	double time_absorption_s = -1.0, time_absorption_v = -1.0;

	using HypreParMatrixPtr = std::unique_ptr<mfem::HypreParMatrix>;
	const mfem::HypreParMatrix *Mtime_s = nullptr, *Mtime_v = nullptr;	
public:
	BlockMomentDiscretization(
		mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &vfes, 
		const BoundaryConditionMap &bc_map, int lumping);
	const mfem::Array<int> &GetOffsets() const { return offsets; }
	virtual mfem::Operator *GetOperator(mfem::Coefficient &total, mfem::Coefficient &absorption) const =0;
	void SetScalarTimeAbsorption(double sigma, const mfem::HypreParMatrix &M);
	void SetVectorTimeAbsorption(double sigma, const mfem::HypreParMatrix &M);
	void SetAlpha(double a) { alpha = a; }

	class Solver : public mfem::Solver {
	private:
		mfem::ParFiniteElementSpace &vfes;
		mfem::Solver &schur_solver;
		
		mfem::Array<int> offsets;
		const mfem::BlockOperator *block_op = nullptr;

		HypreParMatrixPtr iMt, S;
		mutable mfem::Vector t1, t2;
	public:
		Solver(mfem::ParFiniteElementSpace &vfes, mfem::Solver &schur_solver) 
			: vfes(vfes), schur_solver(schur_solver)
		{ }
		void SetOperator(const mfem::Operator &op) override;
		void Mult(const mfem::Vector &b, mfem::Vector &x) const override;
	};
	mfem::HypreParMatrix *FormSchurComplement(const mfem::Operator &op) const;
};

class BlockLDGDiscretization : public BlockMomentDiscretization {
private:
	mfem::Vector beta;
public:
	BlockLDGDiscretization(
		mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &vfes, 
		const BoundaryConditionMap &bc_map, int lumping);
	void SetBeta(const mfem::Vector &b) { beta = b; }
	mfem::BlockOperator *GetOperator(mfem::Coefficient &total, mfem::Coefficient &absorption) const override;

	friend class ConsistentLDGSMMOperator;
};

class BlockIPDiscretization : public BlockMomentDiscretization {
private:
	double kappa = -1.0, mip_val = 0.0;
	bool scale_penalty = false;
public:
	BlockIPDiscretization(
		mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &vfes, 
		const BoundaryConditionMap &bc_map, int lumping);
	void SetKappa(double k) {
		if (k < 0.0) {
			const int p = fes.GetMaxElementOrder();
			kappa = -1.0 * k * pow(p+1,2);
		} else {
			kappa = k;
		}
	}
	void SetPenaltyLowerBound(double lb) { mip_val = lb; }
	void SetScalePenalty(bool use=true) { scale_penalty = use; }
	mfem::BlockOperator *GetOperator(mfem::Coefficient &total, mfem::Coefficient &absorption) const override;

	friend class ConsistentIPSMMOperator;
};

class P1Discretization : public BlockMomentDiscretization {
public:
	P1Discretization(
		mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &vfes, 
		const BoundaryConditionMap &bc_map, int lumping);
	mfem::HypreParMatrix *GetOperator(mfem::Coefficient &total, mfem::Coefficient &absorption) const override;

	friend class ConsistentP1SMMOperator;
};