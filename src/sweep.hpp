#pragma once 

#include "tvector.hpp"
#include "angular_quadrature.hpp"
#include "mfem.hpp"
#include "igraph.h"
#include "phase_coefficient.hpp"
#include "fixup.hpp"

class InverseAdvectionOperator : public mfem::Operator 
{
public:
	// enum for bitset operations to define lumping scheme 
	// 0 = no lumping 
	// 1 = lump mass 
	// 5 = lump mass and faces 
	// 7 = lump everything 
	enum LumpType {
		MASS = 1, 
		GRADIENT = 2, 
		FACE = 4 
	};
private:
	mfem::ParMesh &mesh; 
	mfem::ParFiniteElementSpace &fes; 
	const AngularQuadrature &quad; 
	mfem::GridFunction &total_data; 

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
	// use lumping 
	int lump = 0; 

	bool apply_fixup = false; 
	NegativeFluxFixupOperator *fixup_op = nullptr; 
public:
	InverseAdvectionOperator(mfem::ParFiniteElementSpace &_fes, const AngularQuadrature &_quad, 
		mfem::GridFunction &_total_data, int reflection_bdr_attr=-1, int lump=0); 
	~InverseAdvectionOperator(); 

	void Mult(const mfem::Vector &source, mfem::Vector &psi) const; 
	void AssembleLocalMatrices(); 
	void SetSendBufferSize(int s);
	// allow disabling or re-enabling fixup 
	void UseFixup(bool use=true) { apply_fixup = use; }
	void SetFixupOperator(NegativeFluxFixupOperator &op) {
		fixup_op = &op; 
		apply_fixup = true; 
	}
	void WriteGraphToDot(std::string prefix) const; 

	// lumping type accessors 
	int GetLumpingType() const { return lump; }
	bool IsMassLumped() const { return lump & LumpType::MASS; }
	bool IsGradientLumped() const { return lump & LumpType::GRADIENT; }
	bool IsFaceLumped() const { return lump & LumpType::FACE; }

	friend class AdvectionOperator; 
};

void FormTransportSource(mfem::ParFiniteElementSpace &fes, AngularQuadrature &quad, 
	const mfem::Array<double> &energy_grid, PhaseSpaceCoefficient &source_coef, 
	PhaseSpaceCoefficient &inflow_coef, TransportVectorView source_view); 

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

class AdvectionOperator : public mfem::Operator
{
private:
	const InverseAdvectionOperator &Linv; 
public:
	AdvectionOperator(const InverseAdvectionOperator &Linv) : Linv(Linv) 
	{ }
	void Mult(const mfem::Vector &x, mfem::Vector &y) const override; 
};