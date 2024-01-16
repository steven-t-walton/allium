#pragma once 

#include "tvector.hpp"
#include "angular_quadrature.hpp"
#include "mfem.hpp"
#include "igraph.h"

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
public:
	InverseAdvectionOperator(mfem::ParFiniteElementSpace &_fes, const AngularQuadrature &_quad, 
		const TransportVectorExtents &_psi_ext, mfem::Coefficient &_total, mfem::Coefficient &_inflow); 
	~InverseAdvectionOperator(); 
	void Mult(const mfem::Vector &source, mfem::Vector &psi) const; 

	void WriteGraphToDot(std::string prefix) const; 
};

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

// void FormTransportSourceFromLuaFunctions(mfem::ParFiniteElementSpace &fes, AngularQuadrature &quad, 
// 	const TransportVectorExtents &psi_ext, 
// 	std::function<double(double x, double y, double z, double mu, double eta, double xi)> source_func, 
// 	std::function<double(double x, double y, double z, double mu, double eta, double xi)> inflow_func, 
// 	mfem::Vector &source_vec);

void FormTransportSource(mfem::ParFiniteElementSpace &fes, AngularQuadrature &quad, 
	const TransportVectorExtents &psi_ext, mfem::Coefficient &source, mfem::Coefficient &inflow, mfem::Vector &source_vec); 