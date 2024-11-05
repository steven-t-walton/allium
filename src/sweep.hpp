#pragma once 

#include "bdr_conditions.hpp"
#include "tvector.hpp"
#include "angular_quadrature.hpp"
#include "mfem.hpp"
#include "igraph.h"
#include "phase_coefficient.hpp"
#include "fixup.hpp"

class MultiGroupEnergyGrid; // forward declare to prevent compile dependency
class MultiGroupCoefficient;
class InverseAdvectionOperator : public mfem::Operator 
{
protected:
	mfem::ParMesh &mesh; 
	mfem::ParFiniteElementSpace &fes; 
	const AngularQuadrature &quad; 
	// mfem::GridFunction &total_data; 
	MultiGroupCoefficient &total;

	// multi index extents for angular flux and the "ghost" 
	// buffer psi_fnbr
	TransportVectorExtents psi_ext, psi_fnbr_ext; 

	// this processor owns element ids in the 
	// range [mesh_offsets[0], mesh_offsets[1]) 
	mfem::Array<HYPRE_BigInt> mesh_offsets; 
	// mesh_offsets[0] for each neighboring processor 
	mfem::Array<HYPRE_BigInt> mesh_face_nbr_offsets; 
	// map processor id to face neighbor index 
	std::unordered_map<int,int> proc_to_fn; 
	// map a ghost element id to its global id 
	mfem::Array<HYPRE_BigInt> fnbr_to_global; 
	// "inverse" map: global id to the ghost element id 
	std::unordered_map<HYPRE_BigInt,int> global_to_fnbr; 
	// face neighbor element id to face number 
	mfem::Array<int> fnbr_to_fn; 
	// maps face ids to boundary element ids 
	std::unordered_map<int,int> face_to_bdr_el; 
	// table that maps element ids to face ids 
	// used in sweep to do face integration 
	std::unique_ptr<const mfem::Table> element_to_face; 
	// the local graph object 
	igraph_t graph; 

	// data structures for storing component matrices of sweep 
	mfem::Array<mfem::DenseMatrix*> mass_matrices, grad_matrices, face_matrices;  
	// list of normals on each face 
	mfem::Vector normals; 
	// map an igraph edge id to the mesh's face id 
	mfem::Array<int> edge_to_face_id; 

	// time-dependent sweep data 
	bool is_time_dependent = false; 
	mfem::Array<mfem::DenseMatrix*> time_mass_matrices;
	double time_absorption;	

	// parallel buffer for psi 
	mutable mfem::Vector psi_fnbr; 
	// local buffer for sending messages 
	mutable mfem::Vector par_data_buffer; 
	// storage space for number of incoming edges at each vertex 
	mutable mfem::Array<igraph_integer_t> degrees; 

	// number of elements to sweep before sending a message 
	// packs messages to reduce network saturation 
	// at the expense of making downwind neighbors wait longer 
	// to receive data 
	int send_buffer_size = 8; 
	int max_sends_per_recv = 8;
	// use lumping 
	int lump = 0; 

	bool apply_fixup = false; 
	const NegativeFluxFixupOperator *fixup_op = nullptr; 
	mfem::Vector *fixup_monitor = nullptr;
public:
	InverseAdvectionOperator(mfem::ParFiniteElementSpace &_fes, const AngularQuadrature &_quad, 
		MultiGroupCoefficient &total, const BoundaryConditionMap &bc_map, int lump=0); 
	~InverseAdvectionOperator(); 

	virtual void Mult(const mfem::Vector &source, mfem::Vector &psi) const; 

	// must be called before Mult 
	// pre-assembles components of local sweep matrix 
	// to save cost 
	void AssembleLocalMatrices(); 
	// indicates sweep is time dependent with 
	// provided time absorption (e.g. 1/c/dt)
	// time mass matrix is assembled once and stored
	void SetTimeAbsorption(const double sigma); 
	// access set time absorption value 
	double GetTimeAbsorption() const { return time_absorption; }

	void SetSendBufferSize(int s);
	void SetMaxSendsPerReceive(int s) { max_sends_per_recv = s; }
	// allow disabling or re-enabling fixup 
	void UseFixup(bool use=true);
	bool IsFixupOn() const { return apply_fixup and fixup_op; }
	void SetFixupOperator(const NegativeFluxFixupOperator &op) {
		fixup_op = &op; 
		apply_fixup = true; 
	}
	void SetFixupMonitorData(mfem::Vector &x) { fixup_monitor = &x; }
	void WriteGraphToDot(std::string prefix) const; 
	void WriteGlobalGraphToDot(std::string prefix) const;

	// lumping type accessors 
	int GetLumpingType() const { return lump; }

	friend class AdvectionOperator; 
};

class ParallelBlockJacobiSweepOperator : public InverseAdvectionOperator
{
private:

public:
	ParallelBlockJacobiSweepOperator(mfem::ParFiniteElementSpace &fes, const AngularQuadrature &quad, 
		MultiGroupCoefficient &total, const BoundaryConditionMap &bc_map, int lump)
		: InverseAdvectionOperator(fes, quad, total, bc_map, lump)
	{
		psi_fnbr = 0.0;
	}
	void Mult(const mfem::Vector &source, mfem::Vector &psi) const;
	void Exchange(const mfem::Vector &psi); 
};

void FormTransportSource(mfem::ParFiniteElementSpace &fes, AngularQuadrature &quad, 
	const mfem::Array<double> &energy_grid, PhaseSpaceCoefficient &source_coef, 
	PhaseSpaceCoefficient &inflow_coef, TransportVectorView source_view); 
void FormTransportSource(
	mfem::FiniteElementSpace &fes, const AngularQuadrature &quad,const MultiGroupEnergyGrid &energy_grid, 
	PhaseSpaceCoefficient &source_coef, PhaseSpaceCoefficient &inflow_coef, 
	mfem::Vector &source);

// assemble face mass matrices using the FaceElementTransformations object 
// used in sweep by multiplying the face matrices by Omega.normal 
// and using upwinding 
class FaceMassMatricesIntegrator : public mfem::BilinearFormIntegrator 
{
private:
	mfem::Vector shape1, shape2; 
public:
	void AssembleElementMatrix(const mfem::FiniteElement&, mfem::ElementTransformation&, mfem::DenseMatrix&) {
		MFEM_ABORT("only defined for faces"); 
	}
	void AssembleFaceMatrix(const mfem::FiniteElement &el1, const mfem::FiniteElement &el2, 
		mfem::FaceElementTransformations &trans, mfem::DenseMatrix &elmat);
};

// apply the transport streaming and collision operator 
// performs a halo exchange of all angles, groups 
// uses assembled matrices from InverseAdvectionOperator 
class AdvectionOperator : public mfem::Operator
{
private:
	const InverseAdvectionOperator &Linv; 
public:
	AdvectionOperator(const InverseAdvectionOperator &Linv) : Linv(Linv) 
	{ }
	void Mult(const mfem::Vector &x, mfem::Vector &y) const override; 
	double ComputeBalance(const mfem::Vector &psi) const;
};