#pragma once 

#include "mfem.hpp"

class ConsistentP1Diffusion : public mfem::Operator 
{
private:
	const mfem::ParFiniteElementSpace &fes, &vfes; 
	mfem::Coefficient &total, &scattering; 
public:
	ConsistentP1Diffusion(const mfem::ParFiniteElementSpace &fes, const mfem::ParFiniteElementSpace &vfes, 
		mfem::Coefficient &total, mfem::Coefficient &scattering); 
};