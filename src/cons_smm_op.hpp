#pragma once 
#include "mfem.hpp"
#include "angular_quadrature.hpp"
#include "tvector.hpp"
#include "dg_trace_coll.hpp"
#include "block_smm_op.hpp"

class ConsistentSMMSourceOperatorBase : public mfem::Operator {
protected:
	mfem::ParFiniteElementSpace &fes, &vfes; 
	const AngularQuadrature &quad; 
	const TransportVectorExtents &psi_ext; 
	double alpha; 
	int reflect_bdr_attr; 

	std::unique_ptr<const mfem::Table> element_to_face; 
	std::unordered_map<int,int> face_to_bdr_el; 

	mfem::Array<int> offsets; 
	mfem::Vector Q0, Q1; 
	std::unique_ptr<DGTrace_FECollection> trace_coll; 
	std::unique_ptr<mfem::ParFiniteElementSpace> trace_fes, trace_vfes; 
	mutable mfem::ParGridFunction beta, tensor; 

	mutable mfem::Array<int> marshak_bdr_attrs, reflect_bdr_attrs; 
public:
	ConsistentSMMSourceOperatorBase(mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &vfes, 
		const AngularQuadrature &quad, const TransportVectorExtents &psi_ext, ConstTransportVectorView source_vec, 
		double alpha, int reflect_bdr_attr); 
	virtual void Mult(const mfem::Vector &psi, mfem::Vector &source) const; 
private:
	void ComputeHalfRangeInterfaceTerms(ConstTransportVectorView psi, 
		mfem::ParGridFunction &beta, mfem::ParGridFunction &tensor) const;
};

class ConsistentSMMSourceOperator : public ConsistentSMMSourceOperatorBase {
private:
	using HypreParMatrixPtr = std::unique_ptr<mfem::HypreParMatrix>; 
	HypreParMatrixPtr F11, D, F21, F22;  	
	MomentVectorExtents phi_ext; 
	mutable mfem::Vector moments;
public:
	ConsistentSMMSourceOperator(mfem::ParFiniteElementSpace &_fes, mfem::ParFiniteElementSpace &_vfes, 
		const AngularQuadrature &_quad, const TransportVectorExtents &_psi_ext, ConstTransportVectorView source_vec, 
		double _alpha, int _reflect_bdr_attr=-1); 
	void Mult(const mfem::Vector &psi, mfem::Vector &source) const; 
	const mfem::Array<int> &Offsets() const { return offsets; }
};

class ConsistentLDGSMMSourceOperator : public ConsistentSMMSourceOperatorBase {
private:
	const BlockLDGDiffusionDiscretization &lhs; 

	MomentVectorExtents phi_ext; 
	mutable mfem::Vector moments;

	using HypreParMatrixPtr = std::unique_ptr<mfem::HypreParMatrix>; 
	HypreParMatrixPtr Mt, D, Ma; 
public:
	ConsistentLDGSMMSourceOperator(const BlockLDGDiffusionDiscretization &lhs, 
		const AngularQuadrature &quad, const TransportVectorExtents &_psi_ext, ConstTransportVectorView source_vec); 
	void Mult(const mfem::Vector &psi, mfem::Vector &source) const; 
};

class ConsistentIPSMMSourceOperator : public ConsistentSMMSourceOperatorBase {
private:
	const BlockIPDiffusionDiscretization &lhs; 

	MomentVectorExtents phi_ext; 
	mutable mfem::Vector moments;

	using HypreParMatrixPtr = std::unique_ptr<mfem::HypreParMatrix>; 
	HypreParMatrixPtr Mt, D, Ma; 
public:
	ConsistentIPSMMSourceOperator(const BlockIPDiffusionDiscretization &_lhs, 
		const AngularQuadrature &quad, const TransportVectorExtents &_psi_ext, ConstTransportVectorView source_vec); 
	void Mult(const mfem::Vector &psi, mfem::Vector &source) const; 
};