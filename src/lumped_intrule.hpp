#pragma once 
#include "mfem.hpp"

// sets up a nodal quadrature rule based on a provided finite element 
class LumpedIntegrationRule : public mfem::IntegrationRule
{
public:
	LumpedIntegrationRule(const mfem::FiniteElement &fe);
};
