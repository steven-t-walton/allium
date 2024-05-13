#pragma once 

#include "config.hpp"
#include "mdspan/mdspan.hpp"
#include "mfem.hpp"
#include "phase_coefficient.hpp"

#ifdef TRANSPORT_VECTOR_LAYOUT_LEFT
using TransportVectorLayout = Kokkos::layout_left; 
#else 
#ifdef TRANSPORT_VECTOR_LAYOUT_RIGHT 
using TransportVectorLayout = Kokkos::layout_right; 
#else 
#error "transport vector layout not defined"
#endif
#endif

// index into transport vector with (energy, angle, space) 
using TransportVectorExtents = Kokkos::dextents<std::size_t,3>; 
using TransportVectorView = Kokkos::mdspan<double,TransportVectorExtents,TransportVectorLayout>; 
using ConstTransportVectorView = Kokkos::mdspan<const double,TransportVectorExtents,TransportVectorLayout>; 

// index into moment vector with (energy, moment index, space)
using MomentVectorExtents = Kokkos::dextents<std::size_t,3>;
using MomentVectorView = Kokkos::mdspan<double,MomentVectorExtents,Kokkos::layout_right>;  
using ConstMomentVectorView = Kokkos::mdspan<const double,MomentVectorExtents,Kokkos::layout_right>;

template<typename T, std::size_t... Extents>
T TotalExtent(const Kokkos::extents<T,Extents...> &ext) {
	T size = 1; 
	for (std::size_t r=0; r<ext.rank(); r++) {
		size *= ext.extent(r); 
	}
	return size; 
}

void ProjectPsi(const mfem::FiniteElementSpace &fes, const AngularQuadrature &quad, 
	PhaseSpaceCoefficient &f, TransportVectorView psi); 
void ProjectPsi(const mfem::FiniteElementSpace &fes, const AngularQuadrature &quad, 
	const mfem::Array<double> &energy_grid, PhaseSpaceCoefficient &f, mfem::Vector &psi); 

class SNTimeMassMatrix : public mfem::Operator
{
private:
	const mfem::FiniteElementSpace &fes; 
	const TransportVectorExtents &psi_ext; 
	mfem::Array<mfem::DenseMatrix*> mats; 
public:
	SNTimeMassMatrix(const mfem::FiniteElementSpace &fes, const TransportVectorExtents &ext, bool lump=false);
	~SNTimeMassMatrix() {
		for (auto *ptr : mats) { delete ptr; }
	}
	void Mult(const mfem::Vector &psi, mfem::Vector &Mpsi) const; 
};