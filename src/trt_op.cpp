#include "trt_op.hpp"
#include "block_diag_op.hpp"
#include "constants.hpp"

SNTimeMassMatrix::SNTimeMassMatrix(const mfem::FiniteElementSpace &fes, 
	const TransportVectorExtents &ext, bool lump)
	: fes(fes), psi_ext(ext) 
{
	mats.SetSize(fes.GetNE()); 
	mfem::Array<int> dofs; 
	mfem::MassIntegrator mi; 
	for (int e=0; e<fes.GetNE(); e++) {
		fes.GetElementDofs(e, dofs); 
		mats[e] = new mfem::DenseMatrix(dofs.Size()); 
		mi.AssembleElementMatrix(*fes.GetFE(e), *fes.GetElementTransformation(e), *mats[e]); 
		if (lump)
			mats[e]->Lump(); 
	}
}

void SNTimeMassMatrix::Mult(const mfem::Vector &psi, mfem::Vector &Mpsi) const 
{
	auto psi_view = ConstTransportVectorView(psi.GetData(), psi_ext); 
	auto Mpsi_view = TransportVectorView(Mpsi.GetData(), psi_ext); 
	mfem::Array<int> dofs; 
	mfem::Vector psi_local, Mpsi_local; 
	for (int g=0; g<psi_ext.extent(0); g++) {
		for (int a=0; a<psi_ext.extent(1); a++) {
			for (int e=0; e<fes.GetNE(); e++) {
				fes.GetElementDofs(e, dofs); 
				const auto &elmat = *mats[e]; 
				psi_local.SetSize(dofs.Size()); 
				Mpsi_local.SetSize(dofs.Size()); 					
				for (int n=0; n<dofs.Size(); n++) psi_local(n) = psi_view(g,a,dofs[n]); 
				elmat.Mult(psi_local, Mpsi_local); 
				for (int n=0; n<dofs.Size(); n++) Mpsi_view(g,a,dofs[n]) = Mpsi_local(n); 
			}
		}
	}
}

PicardTRTOperator::PicardTRTOperator(
	const mfem::Array<int> &offsets, 
	const mfem::Operator &Linv, 
	const mfem::Operator &B, 
	const mfem::Operator &C, 
	const mfem::Solver &meb_solver, 
	mfem::Solver &schur_solver)
	: offsets(offsets), Linv(Linv), B(B), C(C), meb_solver(meb_solver), schur_solver(schur_solver)
{
	height = width = offsets.Last(); 
	t1.SetSize(offsets[2] - offsets[1]); // size of temperature 
	t2.SetSize(offsets[2] - offsets[1]); 
}

void PicardTRTOperator::Mult(const mfem::Vector &x, mfem::Vector &y) const 
{
	const mfem::Vector source_psi(*const_cast<mfem::Vector*>(&x), offsets[0], offsets[1] - offsets[0]); 
	const mfem::Vector source_phi(*const_cast<mfem::Vector*>(&x), offsets[1], offsets[2] - offsets[1]); 
	mfem::Vector psi(y, offsets[0], offsets[1] - offsets[0]); 
	mfem::Vector phi(y, offsets[1], offsets[2] - offsets[1]); 

	// forward solve down to temperature problem 
	Linv.Mult(source_psi, psi); 
	C.Mult(psi, t1); 
	add(source_phi, 1.0, t1, t1); // source_phi - t1 -> t1 

	// solve schur complement in temperature 
	SchurComplementOperator schur(Linv, B, C, meb_solver, t1, t2, psi); 
	schur_solver.SetOperator(schur); 
	mfem::Vector blank;
	schur_solver.Mult(blank, phi); 

	// back solve to get psi 
	B.Mult(phi, psi); 
	add(source_psi, 1.0, psi, psi); // source_psi - psi -> psi 
	Linv.Mult(psi, psi); 
}

void PicardTRTOperator::SchurComplementOperator::Mult(const mfem::Vector &x, mfem::Vector &y) const 
{
	B.Mult(x, psi);
	Linv.Mult(psi, psi); 
	C.Mult(psi, tmp); 
	add(tmp, 1.0, source, tmp); 
	meb_solver.Mult(tmp, y); 
}