#include "p1diffusion.hpp"

ConsistentP1Diffusion::ConsistentP1Diffusion(const mfem::ParFiniteElementSpace &_fes, const mfem::ParFiniteElementSpace &_vfes,
	mfem::Coefficient &_total, mfem::Coefficient &_scattering)
	: fes(_fes), vfes(_vfes), total(_total), scattering(scattering)
{

}