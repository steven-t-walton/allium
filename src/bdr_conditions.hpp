#pragma once 

#include "mfem.hpp"

enum BoundaryCondition {
	INFLOW, 
	REFLECTIVE
};

using BoundaryConditionMap = std::unordered_map<int,BoundaryCondition>;

template<BoundaryCondition BC>
mfem::Array<int> CreateBdrAttributeMarker(const BoundaryConditionMap &bc_map) {
	// compute maximum attribute
	// for list size 
	int v = 0;
	for (const auto &bc : bc_map) {
		v = std::max(bc.first, v);
	}

	// mark attributes of type BC 
	mfem::Array<int> list(v);
	list = 0;
	for (const auto &bc : bc_map) {
		if (bc.second == BC) {
			list[bc.first - 1] = 1;
		}
	}
	return list;
}
