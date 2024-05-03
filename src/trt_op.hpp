#pragma once 

#include "mfem.hpp"
#include "tvector.hpp"

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