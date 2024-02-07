#pragma once 

#include "tvector.hpp"
#include "angular_quadrature.hpp"
#include "mfem.hpp"
#include "igraph.h"
#include "phase_coefficient.hpp"

class InverseAdvectionOperator : public mfem::Operator 
{
private:
	mfem::ParMesh &mesh; 
	mfem::ParFiniteElementSpace &fes; 
	const AngularQuadrature &quad; 
	const TransportVectorExtents &psi_ext; 
	mfem::Coefficient &total, &inflow; 

	// this processor owns element ids in the 
	// range [mesh_offsets[0], mesh_offsets[1]) 
	mfem::Array<HYPRE_BigInt> mesh_offsets; 
	mfem::Array<HYPRE_BigInt> dof_offsets; 
	mfem::Array<HYPRE_BigInt> mesh_face_nbr_offsets; 
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
	bool exchange_downwind = false; 
	mfem::Table downwind_send_table; 
	mfem::Table downwind_recv_table; 

	mfem::Array<mfem::DenseMatrix*> mass_matrices, grad_matrices, face_matrices;  

	mutable mfem::Vector psi_fnbr; 

	// number of elements to sweep before sending a message 
	// packs messages to reduce network saturation 
	// at the expense of making downwind neighbors wait longer 
	// to receive data 
	int send_buffer_size = 8; 
public:
	InverseAdvectionOperator(mfem::ParFiniteElementSpace &_fes, const AngularQuadrature &_quad, 
		const TransportVectorExtents &_psi_ext, mfem::Coefficient &_total, mfem::Coefficient &_inflow); 
	~InverseAdvectionOperator(); 
	void Mult(const mfem::Vector &source, mfem::Vector &psi) const; 

	void SetSendBufferSize(int s); 
	void WriteGraphToDot(std::string prefix) const; 
};

void FormTransportSource(mfem::ParFiniteElementSpace &fes, AngularQuadrature &quad, 
	PhaseSpaceCoefficient &source_coef, PhaseSpaceCoefficient &inflow_coef, 
	TransportVectorView source_view); 

void FormTransportSource(mfem::ParFiniteElementSpace &fes, AngularQuadrature &quad, 
	const TransportVectorExtents &psi_ext,
	std::function<double(const mfem::Vector &x, const mfem::Vector &y)> source_func, 
	std::function<double(const mfem::Vector &x, const mfem::Vector &y)> inflow_func, 
	mfem::Vector &source_vec); 

void FormTransportSource(mfem::ParFiniteElementSpace &fes, AngularQuadrature &quad, 
	const TransportVectorExtents &psi_ext,
	std::function<double(double x, double y, double z, double mu, double eta, double xi)> source_func, 
	std::function<double(double x, double y, double z, double mu, double eta, double xi)> inflow_func, 
	mfem::Vector &source_vec); 

void FormTransportSource(mfem::ParFiniteElementSpace &fes, AngularQuadrature &quad, 
	const TransportVectorExtents &psi_ext, mfem::Coefficient &source, mfem::Coefficient &inflow, mfem::Vector &source_vec); 

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