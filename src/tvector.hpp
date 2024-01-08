#pragma once 

#include "config.hpp"
#include "mdspan/mdspan.hpp"

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

// index into moment vector with (moment number, energy, space)
using MomentVectorExtents = Kokkos::dextents<std::size_t,3>;
using MomentVectorView = Kokkos::mdspan<double,MomentVectorExtents,Kokkos::layout_right>;  

template<typename T, std::size_t... Extents>
T TotalExtent(const Kokkos::extents<T,Extents...> &ext) {
	T size = 1; 
	for (std::size_t r=0; r<ext.rank(); r++) {
		size *= ext.extent(r); 
	}
	return size; 
}
