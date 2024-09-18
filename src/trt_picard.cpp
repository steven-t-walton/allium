#include "trt_picard.hpp"
#include "mg_form.hpp"

PicardTRTOperator::PicardTRTOperator(
	const mfem::Array<int> &offsets, 
	InverseAdvectionOperator &Linv, 
	const mfem::Operator &D, 
	const mfem::Operator &emission, 
	const mfem::Operator &sigma, 
	const mfem::Solver &meb_solver,
	mfem::Solver &schur_solver)
	: offsets(offsets), Linv(Linv), meb_solver(meb_solver), schur_solver(schur_solver),
	  DT(D), B(&DT, &emission, false, false), C(&sigma, &D, false, false)
{
	height = width = offsets.Last(); 
	tmp.SetSize(offsets[2] - offsets[1]); // size of temperature 
}

void PicardTRTOperator::Mult(const mfem::Vector &x, mfem::Vector &y) const 
{
	const mfem::Vector source_psi(*const_cast<mfem::Vector*>(&x), offsets[0], offsets[1] - offsets[0]); 
	const mfem::Vector source_T(*const_cast<mfem::Vector*>(&x), offsets[1], offsets[2] - offsets[1]); 
	mfem::Vector psi(y, offsets[0], offsets[1] - offsets[0]); 
	mfem::Vector T(y, offsets[1], offsets[2] - offsets[1]); 

	B.Mult(T, psi);
	psi += source_psi;
	Linv.Mult(psi, psi);
	C.Mult(psi, tmp);
	tmp += source_T;
	meb_solver.Mult(tmp, T);

	if (opacity) {
		opacity->Project();
		Mtot->Assemble();
		Mtot->Finalize();
		Linv.AssembleLocalMatrices();
	}
}