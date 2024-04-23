#include "lumped_intrule.hpp"

LumpedIntegrationRule::LumpedIntegrationRule(const mfem::FiniteElement &fe)
{
	const auto &nodes = fe.GetNodes(); 
	SetSize(nodes.Size()); 
	double w = 1.0/nodes.Size(); 
	for (int i=0; i<Size(); i++) {
		(*this)[i] = nodes[i]; 
		(*this)[i].weight = w; 
	}
}
