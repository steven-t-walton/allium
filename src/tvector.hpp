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

// arbitrary flattening of 3D index into 1D 
// for index operator(i,j,k)
// I controls the stride ordering of the first index
// J the second, and K the third 
// IndexMap<1,0,2>: strides fastest in argument 1 (j), 
// then argument 0 (i), then argument 2 (k) 
template<std::size_t I,std::size_t J, std::size_t K>
struct IndexMap
{
	static_assert(I >= 0 and I <= 2); static_assert(J >= 0 and J <= 2); static_assert(K >= 0 and K <= 2);
	static_assert(I+J+K == 3);
	template <class Extents>
	class mapping
	{
	private:
		using index_type = typename Extents::index_type;
		Extents ext;
	public:
		mapping(const Extents &ext) : ext(ext) {
			static_assert(ext.rank()==3);
		}
		template<class... Indices>
		constexpr index_type operator()(Indices... idxs) const noexcept {
			std::array<std::common_type_t<Indices...>,sizeof...(idxs)> idx{idxs...};
			return idx[I] + ext.extent(I)*(idx[J] + ext.extent(J)*idx[K]);
		}
		constexpr const Extents& extents() const noexcept
		{
			return ext;
		}
	};	
};

// index into transport vector with (energy, angle, space) 
struct TransportIndex {
	static constexpr int ENERGY = 0;
	static constexpr int ANGLE = 1;
	static constexpr int SPACE = 2;
};
using TransportVectorExtents = Kokkos::dextents<std::size_t,3>; 
using TransportVectorView = Kokkos::mdspan<double,TransportVectorExtents,TransportVectorLayout>; 
using ConstTransportVectorView = Kokkos::mdspan<const double,TransportVectorExtents,TransportVectorLayout>; 

// index into moment vector with (energy, moment index, space)
struct MomentIndex {
	static constexpr int ENERGY = 0;
	static constexpr int MOMENT = 1; 
	static constexpr int SPACE = 2;
};
using MomentVectorExtents = Kokkos::dextents<std::size_t,3>;
// strides fastest in space, then energy, then moment id
using MomentVectorLayout = IndexMap<MomentIndex::SPACE,MomentIndex::ENERGY,MomentIndex::MOMENT>; 
using MomentVectorView = Kokkos::mdspan<double,MomentVectorExtents,MomentVectorLayout>;  
using ConstMomentVectorView = Kokkos::mdspan<const double,MomentVectorExtents,MomentVectorLayout>;

// evaluate group-wise dependence of the zeroth moment 
class ZerothMomentCoefficient : public mfem::VectorCoefficient {
private:
	const mfem::FiniteElementSpace &fes;
	const MomentVectorExtents &phi_ext;
	const mfem::Vector &data;
	const int moment_id = 0;
	mfem::Vector shape, local_data;
public:
	ZerothMomentCoefficient(
		const mfem::FiniteElementSpace &fes, const MomentVectorExtents &phi_ext, const mfem::Vector &data);
	void Eval(mfem::Vector &v, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) override;
};

// evaluate group-wise dependence of the first moment
// return matrix is (num groups) x (dim components of F)
class FirstMomentCoefficient : public mfem::MatrixCoefficient {
private:
	const mfem::FiniteElementSpace &fes; 
	const MomentVectorExtents &ext;
	const mfem::Vector &data; 
	mfem::Vector shape; 
public:
	FirstMomentCoefficient(const mfem::FiniteElementSpace &fes, const MomentVectorExtents &ext, 
		const mfem::Vector &data);
	void Eval(mfem::DenseMatrix &K, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) override;
};

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

// forward declare to avoid circular dependency
class MultiGroupEnergyGrid;
void ProjectPsi(const mfem::FiniteElementSpace &fes, const AngularQuadrature &quad, 
	const MultiGroupEnergyGrid &energy_grid, PhaseSpaceCoefficient &f, mfem::Vector &psi);

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