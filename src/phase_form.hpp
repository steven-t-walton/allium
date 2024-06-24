#pragma once 

#include "mfem.hpp"
#include "tvector.hpp"

// assemble into a moment vector-sized nonlinear form 
class MomentVectorNonlinearForm : public mfem::NonlinearForm 
{
private:
	const TransportVectorExtents &phi_ext;
	mutable mfem::BlockOperator *grad = nullptr;
public:
	MomentVectorNonlinearForm(mfem::FiniteElementSpace *fes, const TransportVectorExtents &phi_ext) 
		: mfem::NonlinearForm(fes), phi_ext(phi_ext)
	{
		// overwrite height for group output
		const auto G = phi_ext.extent(MomentIndex::ENERGY);
		const auto M = phi_ext.extent(MomentIndex::MOMENT);
		if (M != 1) MFEM_ABORT("moments not implemented yet");
		height *= G * M;
	}
	~MomentVectorNonlinearForm() { delete grad; }
	void Mult(const mfem::Vector &x, mfem::Vector &y) const override;
	// mfem::BlockOperatorperator &GetGradient(const mfem::Vector &x) const override;
};