#pragma once 
#include "mfem.hpp"

// enum for bitset operations to define lumping scheme 
// 0 = no lumping 
// 1 = lump mass 
// 5 = lump mass and faces 
// 7 = lump everything 
enum LumpingType {
	MASS = 1, 
	GRADIENT = 2, 
	FACE = 4 
};

// sets up a nodal quadrature rule based on a provided finite element 
class LumpedIntegrationRule : public mfem::IntegrationRule
{
public:
	LumpedIntegrationRule(const mfem::FiniteElement &fe);
};
