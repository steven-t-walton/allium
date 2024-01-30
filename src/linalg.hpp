#pragma once 

#include "mfem.hpp"

mfem::HypreParMatrix *ElementByElementBlockInverse(const mfem::ParFiniteElementSpace &fes, const mfem::HypreParMatrix &A); 

class ComponentExtractionOperator : public mfem::Operator 
{
private:
	mfem::Array<int> offsets; 
	int comp; 
public:
	ComponentExtractionOperator(const mfem::Array<int> &_offsets, int c) 
		: offsets(_offsets), comp(c) {
			height = offsets[comp+1] - offsets[comp]; 
			width = offsets.Last(); 
		}
	void Mult(const mfem::Vector &x, mfem::Vector &y) const {
		mfem::BlockVector bx(*const_cast<mfem::Vector*>(&x), 0, offsets); 
		y = bx.GetBlock(comp);
	}
	void MultTranspose(const mfem::Vector &x, mfem::Vector &y) const {
		y = 0.0; 
		mfem::BlockVector by(y.GetData(), offsets); 
		by.GetBlock(comp) = x; 
	}
};