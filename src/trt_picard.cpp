#include "trt_picard.hpp"

PicardTRTOperator::PicardTRTOperator(
	const mfem::Array<int> &offsets, 
	const mfem::Operator &Linv, 
	const mfem::Operator &D, 
	const mfem::Operator &emission, 
	const mfem::Operator &sigma, 
	const mfem::Solver &meb_solver,
	mfem::Solver &schur_solver)
	: offsets(offsets), Linv(Linv), meb_solver(meb_solver), schur_solver(schur_solver),
	  DT(D), B(&DT, &emission, false, false), C(&sigma, &D, false, false)
{
	height = width = offsets.Last(); 
	t1.SetSize(offsets[2] - offsets[1]); // size of temperature 
	t2.SetSize(offsets[2] - offsets[1]); 
}

void PicardTRTOperator::Mult(const mfem::Vector &x, mfem::Vector &y) const 
{
	const mfem::Vector source_psi(*const_cast<mfem::Vector*>(&x), offsets[0], offsets[1] - offsets[0]); 
	const mfem::Vector source_T(*const_cast<mfem::Vector*>(&x), offsets[1], offsets[2] - offsets[1]); 
	mfem::Vector psi(y, offsets[0], offsets[1] - offsets[0]); 
	mfem::Vector T(y, offsets[1], offsets[2] - offsets[1]); 

	// forward solve down to temperature problem 
	Linv.Mult(source_psi, psi); 
	C.Mult(psi, t1); 
	add(source_T, 1.0, t1, t1); // source_T - t1 -> t1 

	// solve schur complement in temperature 
	FixedPointOperator schur(Linv, B, C, meb_solver, t1, t2, psi); 
	schur_solver.SetOperator(schur); 
	mfem::Vector blank;
	schur_solver.Mult(blank, T); 

	// back solve to get psi 
	B.Mult(T, psi); 
	add(source_psi, 1.0, psi, psi); // source_psi - psi -> psi 
	Linv.Mult(psi, psi); 
}

void PicardTRTOperator::
FixedPointOperator::Mult(const mfem::Vector &x, mfem::Vector &y) const 
{
	B.Mult(x, psi);
	Linv.Mult(psi, psi); 
	C.Mult(psi, tmp); 
	add(tmp, 1.0, source, tmp); 
	meb_solver.Mult(tmp, y); 
}