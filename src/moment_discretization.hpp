#pragma once 

#include "mfem.hpp"
#include "bdr_conditions.hpp"
#include "block_diag_op.hpp"

class MomentDiscretization
{
protected:
	mfem::ParFiniteElementSpace &fes; 
	mfem::Coefficient &total, &absorption;
	const BoundaryConditionMap &bc_map;
	int lumping;

	mutable mfem::Array<int> marshak_bdr_attrs, reflect_bdr_attrs;
	double alpha = 0.5; 
	double time_absorption = -1.0;

	using HypreParMatrixPtr = std::unique_ptr<mfem::HypreParMatrix>;
	HypreParMatrixPtr Mtime;
public:
	MomentDiscretization(mfem::ParFiniteElementSpace &fes, 
		mfem::Coefficient &total, mfem::Coefficient &absorption, 
		const BoundaryConditionMap &bc_map, int lumping);
	virtual mfem::HypreParMatrix *GetOperator() const =0;

	void SetAlpha(double a) { alpha = a; }
	void SetTimeAbsorption(double sigma);
};

class H1DiffusionDiscretization : public MomentDiscretization
{
public:
	H1DiffusionDiscretization(mfem::ParFiniteElementSpace &fes, 
		mfem::Coefficient &total, mfem::Coefficient &absorption, 
		const BoundaryConditionMap &bc_map, int lumping);
	mfem::HypreParMatrix *GetOperator() const override;	
};

class InteriorPenaltyDiscretization : public MomentDiscretization 
{
private:
	double kappa = -1.0, mip_val = 0.0, sigma = -1.0;
public:
	InteriorPenaltyDiscretization(mfem::ParFiniteElementSpace &fes, 
		mfem::Coefficient &total, mfem::Coefficient &absorption,
		const BoundaryConditionMap &bc_map, int lumping);
	mfem::HypreParMatrix *GetOperator() const override;

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
	LDGDiscretization(mfem::ParFiniteElementSpace &fes, 
		mfem::Coefficient &total, mfem::Coefficient &absorption, 
		const BoundaryConditionMap &bc_map, int lumping);
	void SetBeta(const mfem::Vector &b) { beta = b; } 
	mfem::HypreParMatrix *GetOperator() const override;
};

class InverseMomentDiscretization
{
private:
	const MomentDiscretization &disc;
	mfem::Solver &solver;

	std::unique_ptr<mfem::HypreParMatrix> oper;
public:
	InverseMomentDiscretization(const MomentDiscretization &disc, mfem::Solver &solver)
		: disc(disc), solver(solver)
	{ }
	const mfem::Solver &GetSolver()
	{
		oper.reset(disc.GetOperator());
		solver.SetOperator(*oper);
		return solver;
	}
};

class BlockMomentDiscretization {
protected:
	mfem::ParFiniteElementSpace &fes, &vfes; 
	mfem::Coefficient &total, &absorption;
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
		mfem::Coefficient &total, mfem::Coefficient &absorption,
		const BoundaryConditionMap &bc_map, int lumping);
	const mfem::Array<int> &GetOffsets() const { return offsets; }
	virtual mfem::BlockOperator *GetOperator() const =0;
	void SetScalarTimeAbsorption(double sigma, const mfem::HypreParMatrix &M);
	void SetVectorTimeAbsorption(double sigma, const mfem::HypreParMatrix &M);
	void SetAlpha(double a) { alpha = a; }

	mfem::Coefficient &GetTotal() { return total; }
	mfem::Coefficient &GetAbsorption() { return absorption; }

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
		mfem::Coefficient &total, mfem::Coefficient &absorption,
		const BoundaryConditionMap &bc_map, int lumping);
	void SetBeta(const mfem::Vector &b) { beta = b; }
	mfem::BlockOperator *GetOperator() const override;

	friend class ConsistentLDGSMMOperator;
};

class BlockIPDiscretization : public BlockMomentDiscretization {
private:
	double kappa = -1.0, mip_val = 0.0;
	bool scale_penalty = false;
public:
	BlockIPDiscretization(
		mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &vfes, 
		mfem::Coefficient &total, mfem::Coefficient &absorption,
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
	mfem::BlockOperator *GetOperator() const override;

	friend class ConsistentIPSMMOperator;
};

class P1Discretization : public BlockMomentDiscretization {
public:
	P1Discretization(
		mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &vfes, 
		mfem::Coefficient &total, mfem::Coefficient &absorption,
		const BoundaryConditionMap &bc_map, int lumping);
	mfem::BlockOperator *GetOperator() const override;

	friend class ConsistentP1SMMOperator;
};

class RTDiffusionDiscretization : public BlockMomentDiscretization {
private:
	mfem::Array<int> ess_tdof_list;
public:
	RTDiffusionDiscretization(mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &vfes, 
		mfem::Coefficient &total, mfem::Coefficient &absorption, 
		const BoundaryConditionMap &bc_map, int lumping);
	mfem::BlockOperator *GetOperator() const override;	
};

class HybridizedRTDiffusionDiscretization : public BlockMomentDiscretization {
private:
	mfem::ParFiniteElementSpace &ifes;
	mfem::Array<int> br_tdof_marker;
	mfem::Table rt_br_dofs;
public:
	HybridizedRTDiffusionDiscretization(
		mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &vfes, 
		mfem::ParFiniteElementSpace &ifes, mfem::Coefficient &total, 
		mfem::Coefficient &absorption, const BoundaryConditionMap &bc_map, 
		int lumping);
	mfem::BlockOperator *GetOperator() const override;

	class HatSolver : public mfem::Solver
	{
	private:
		mfem::ParFiniteElementSpace &ifes;
		mfem::Solver &schur_solver;

		mfem::Array<int> offsets;
		const mfem::BlockOperator *block_op = nullptr;
		HypreParMatrixPtr H;
		std::unique_ptr<mfem::BlockOperator> Ainv;
	public:
		HatSolver(mfem::ParFiniteElementSpace &ifes, mfem::Solver &schur_solver)
			: ifes(ifes), schur_solver(schur_solver)
		{ }
		void SetOperator(const mfem::Operator &op) override;
		void Mult(const mfem::Vector &b, mfem::Vector &x) const override;
	};

	// maps [J,phi] -> [Jhat, phi, lambda] with Mult 
	// and [Jhat, phi, lambda] -> [J,phi] with MultTranspose
	class ProlongationOperator : public mfem::Operator
	{
	private:
		mfem::ParFiniteElementSpace &vfes;
		const mfem::Table &rt_br_dofs;
		mfem::Array<int> row_offsets, col_offsets;
	public:
		ProlongationOperator(const HybridizedRTDiffusionDiscretization &disc)
			: vfes(disc.vfes), rt_br_dofs(disc.rt_br_dofs)
		{
			col_offsets.SetSize(3);
			col_offsets[0] = 0;
			col_offsets[1] = disc.vfes.GetVSize();
			col_offsets[2] = disc.fes.GetVSize();
			col_offsets.PartialSum();

			row_offsets.SetSize(4);
			row_offsets[0] = 0;
			row_offsets[1] = disc.rt_br_dofs.Size_of_connections();
			row_offsets[2] = disc.fes.GetVSize();
			row_offsets[3] = disc.ifes.GetVSize();
			row_offsets.PartialSum();

			height = row_offsets.Last();
			width = col_offsets.Last();
		}
		const mfem::Array<int> &RowOffsets() const { return row_offsets; }
		const mfem::Array<int> &ColOffsets() const { return col_offsets; }
		void Mult(const mfem::Vector &x, mfem::Vector &y) const override;
		void MultTranspose(const mfem::Vector &x, mfem::Vector &y) const override;
	};

	class Solver : public mfem::Solver
	{
	private:
		HatSolver solver;
		ProlongationOperator P;
		std::unique_ptr<mfem::RAPOperator> rap;
	public:
		Solver(const HybridizedRTDiffusionDiscretization &disc, mfem::Solver &schur_solver)
			: solver(disc.ifes, schur_solver), P(disc)
		{
		}
		const mfem::Array<int> &GetOffsets() const { return P.ColOffsets(); }
		void SetOperator(const mfem::Operator &op) override
		{
			solver.SetOperator(op);
			rap = std::make_unique<mfem::RAPOperator>(P, solver, P);
			height = width = rap->Height();
		}
		void Mult(const mfem::Vector &b, mfem::Vector &x) const override
		{
			rap->Mult(b, x);
		}
	};
};