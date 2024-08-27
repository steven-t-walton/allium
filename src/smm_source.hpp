#pragma once 

#include "moment_discretization.hpp"
#include "angular_quadrature.hpp"
#include "tvector.hpp"

class MomentFaceClosuresOperator : public mfem::Operator
{
private:
	mfem::ParFiniteElementSpace &fes, &trace_fes, &trace_vfes;
	const AngularQuadrature &quad;
	const TransportVectorExtents &psi_ext;
	const BoundaryConditionMap &bc_map;

	std::unique_ptr<const mfem::Table> element_to_face; 
	std::unordered_map<int,int> face_to_bdr_el; 
public:
	MomentFaceClosuresOperator(mfem::ParFiniteElementSpace &fes, 
		mfem::ParFiniteElementSpace &trace_fes, mfem::ParFiniteElementSpace &trace_vfes, 
		const AngularQuadrature &quad, const TransportVectorExtents &psi_ext, const BoundaryConditionMap &bc_map);
	void Mult(const mfem::Vector &psi, mfem::Vector &closures) const override;
};

class ConsistentSMMOperatorBase : public mfem::Operator {
protected:
	mfem::ParFiniteElementSpace &fes, &vfes;
	const AngularQuadrature &quad;
	const TransportVectorExtents &psi_ext;
	double alpha; 
	int lumping;
	const BoundaryConditionMap &bc_map;

	mfem::Vector Q, Q0, Q1;
	std::unique_ptr<mfem::FiniteElementCollection> trace_coll; 
	std::unique_ptr<mfem::ParFiniteElementSpace> trace_fes, trace_vfes; 
	mutable mfem::Vector face_closure_data;
	std::unique_ptr<MomentFaceClosuresOperator> face_closure_op;
	// references into face_closure_data
	// persistent storage for ExchangeFaceNbr call 
	mutable mfem::ParGridFunction beta, tensor;

	mutable mfem::Array<int> marshak_bdr_attrs, reflect_bdr_attrs; 
	MomentVectorExtents phi_ext; 
public:
	ConsistentSMMOperatorBase(mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &vfes, 
		const AngularQuadrature &quad, const TransportVectorExtents &psi_ext, const BoundaryConditionMap &bc_map, 
		const mfem::Vector &source_vec, double alpha, int lumping);
	void Mult(const mfem::Vector &psi, mfem::Vector &source) const override;
};

class ConsistentP1SMMOperator : public ConsistentSMMOperatorBase {
private:
	using HypreParMatrixPtr = std::unique_ptr<mfem::HypreParMatrix>; 
	HypreParMatrixPtr F11, D, F21, F22;  	
	mutable mfem::Vector moments;
public:
	ConsistentP1SMMOperator(const P1Discretization &disc,
		const AngularQuadrature &quad, const TransportVectorExtents &psi_ext, const mfem::Vector &source_vec); 
	void Mult(const mfem::Vector &psi, mfem::Vector &source) const override;
};

class ConsistentLDGSMMOperator : public ConsistentSMMOperatorBase {
private:
	mfem::Vector beta;
	mutable mfem::Vector moments;
	using HypreParMatrixPtr = std::unique_ptr<mfem::HypreParMatrix>; 
	HypreParMatrixPtr D, Ma, DT; 
public:
	ConsistentLDGSMMOperator(const BlockLDGDiscretization &disc, 
		const AngularQuadrature &quad, const TransportVectorExtents &psi_ext, const mfem::Vector &source_vec);
	void Mult(const mfem::Vector &psi, mfem::Vector &source) const override;
};

class ConsistentIPSMMOperator : public ConsistentSMMOperatorBase {
private:
	mfem::Coefficient &total;
	double kappa, mip_val; 
	bool scale_penalty;

	mutable mfem::Vector moments;

	using HypreParMatrixPtr = std::unique_ptr<mfem::HypreParMatrix>;
	HypreParMatrixPtr Ma, D, DT;
public:
	ConsistentIPSMMOperator(const BlockIPDiscretization &disc, mfem::Coefficient &total, 
		const AngularQuadrature &quad, const TransportVectorExtents &psi_ext, const mfem::Vector &source_vec);
	void Mult(const mfem::Vector &psi, mfem::Vector &source) const override;
};

class IndependentSMMOperator : public mfem::Operator {
private:
	mfem::ParFiniteElementSpace &fes, &vfes; 
	const AngularQuadrature &quad; 
	const TransportVectorExtents &psi_ext;
	PhaseSpaceCoefficient &source_coef, &inflow_coef; 
	double alpha; 
	const BoundaryConditionMap &bc_map; 
	int lumping;

	mfem::Vector Q0, Q1;
	mutable mfem::Array<int> marshak_bdr_attrs, reflect_bdr_attrs;
public:
	IndependentSMMOperator(mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &vfes, 
		const AngularQuadrature &quad, const TransportVectorExtents &psi_ext, 
		PhaseSpaceCoefficient &source_coef, PhaseSpaceCoefficient &inflow_coef, 
		double alpha, const BoundaryConditionMap &bc_map, int lumping);
	IndependentSMMOperator(mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &vfes, 
		const AngularQuadrature &quad, const MultiGroupEnergyGrid &energy, 
		const TransportVectorExtents &psi_ext, 
		PhaseSpaceCoefficient &source_coef, PhaseSpaceCoefficient &inflow_coef, 
		double alpha, const BoundaryConditionMap &bc_map, int lumping);
	void Mult(const mfem::Vector &psi, mfem::Vector &source) const override;
};