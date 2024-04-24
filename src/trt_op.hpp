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

class NonlinearFormBlockInverse : public mfem::Operator 
{
private:
	const mfem::NonlinearForm &nform; 
	const mfem::Vector &data; 
	bool lump; 
public:
	NonlinearFormBlockInverse(const mfem::NonlinearForm &nf, const mfem::Vector &d, bool lump=false) 
		: nform(nf), data(d), lump(lump)
	{
		height = width = nform.FESpace()->GetVSize(); 
	}
	// apply inverse of local gradient assembled using nonlinear data in data 
	void Mult(const mfem::Vector &x, mfem::Vector &y) const override;
};