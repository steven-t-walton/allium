#include "trt_op.hpp"
#include "block_diag_op.hpp"

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

LocalEliminationTRTOperator::LocalEliminationTRTOperator(
	const mfem::Operator &Linv, 
	const mfem::Operator &D, 
	const mfem::Operator &emission_form, 
	const mfem::Operator &Mtot,
	const mfem::IterativeSolver &meb_solver, 
	mfem::Vector &psi)
	: Linv(Linv), D(D), emission_form(emission_form), Mtot(Mtot), meb_solver(meb_solver), psi(psi)
{
	height = width = meb_solver.Height(); 
	tmp.SetSize(height); 
}

void LocalEliminationTRTOperator::Mult(const mfem::Vector &x, mfem::Vector &y) const 
{
	emission_form.Mult(x, tmp); 
	D.MultTranspose(tmp, psi); 
	Linv.Mult(psi, psi); 
	D.Mult(psi, y); 
	Mtot.Mult(y, tmp); 
	add(tmp, 1.0, *source, tmp); 
	y = x; 
	meb_solver.Mult(tmp, y); 
}

LinearEliminationTDOperator::LinearEliminationTDOperator(
	const mfem::Array<int> &offsets,
	const mfem::Operator &Linv, 
	const mfem::Operator &D, 
	const mfem::Operator &emission_form, 
	const mfem::Operator &Mtot, 
	const mfem::IterativeSolver &schur_solver)
	: offsets(offsets), Linv(Linv), D(D), emission_form(emission_form), Mtot(Mtot), schur_solver(schur_solver)
{
	height = width = offsets.Last(); 
	type = IMPLICIT; 
}

void LinearEliminationTDOperator::ImplicitSolve(const double dt, const mfem::Vector &x, mfem::Vector &y)
{
	// Mpsi.Mult(psi0, psi0); // operator designed to work in-place 
	// add(source_vec, 1.0/time_step/constants::SpeedOfLight, psi0, psi0); // source_vec + 1/c/dt psi0 -> psi0
	// Mcv.Mult(T, T0); // assume T = T0 to get Mcv T0 -> T0 
	// T0 *= 1.0/time_step; 

	// Linv.Mult(psi0, psi); 
	// D.Mult(psi, phi); 
	// Mtot.Mult(phi, phi_source); 
	// add(T0, 1.0, phi_source, em_source); 
	// op.SetSource(em_source); 
	// mfem::Vector blank; 
	// outer_solver->Mult(blank, T); 
	// emission_form.Mult(T, em_source); 
	// D.MultTranspose(em_source, psi); 
	// psi += psi0; 	
	// Linv.Mult(psi, psi); 
	// D.Mult(psi, phi); 
}

// LinearizedTRTOperator::LinearizedTRTOperator(
// 	const mfem::Array<int> &offsets, 
// 	const InverseAdvectionOperator &Linv, 
// 	const DiscreteToMoment &D, 
// 	const mfem::NonlinearForm &meb_form, 
// 	const mfem::NonlinearForm &emission_form, 
// 	const mfem::Operator &Mtot, 
// 	mfem::IterativeSolver &schur_solver, 
// 	mfem::IterativeSolver *dsa_solver)
// 	: offsets(offsets), Linv(Linv), D(D), meb_form(meb_form), emission_form(emission_form), 
// 		schur_solver(schur_solver), Mtot(Mtot), dsa_solver(dsa_solver)
// {
// 	height = width = offsets.Last(); 
// }

// void LinearizedTRTOperator::Mult(const mfem::Vector &b, mfem::Vector &x) const 
// {

// }

// void LinearizedTRTOperator::SolveJacobian(mfem::Vector &x) const 
// {
// 	mfem::BlockVector 
// 	mfem::Vector psi(x, offsets[0], offsets[1] - offsets[0]); // reference to psi 
// 	mfem::Vector T(x, offsets[1], offsets[2] - offsets[1]); // reference to temperature 

// 	BlockDiagonalByElementSolver meb_grad_inv(Linv.GetLumpingType());

// 	// invert (cv/dt + 4a sigma T^3)
// 	const auto &meb_grad = meb_form.GetGradient(T); 
// 	meb_grad_inv.SetOperator(meb_grad); 
// 	// ( 4a sigma T^3 u, v)
// 	const auto &dplanck = emission_form.GetGradient(T); 
// 	mfem::ProductOperator linearized_elim(&dplanck, &meb_grad_inv, false, false); 

// 	// form transport residual 
// 	emission_form.Mult(T, em_source); 
// 	// eliminate down to phi
// 	linearized_elim.Mult(temp_resid, phi_source);
// 	em_source += phi_source; 
// 	D.MultTranspose(em_source, psi); 
// 	psi += psi0; 
// 	Linv.Mult(psi, psi);
// 	D.Mult(psi, phi_source); 

// 	// apply I - D Linv dB (dBt)^{-1} Msigma 
// 	mfem::TripleProductOperator Ms_form(&dplanck, &meb_grad_inv, &Mtot, false, false, false); 
// 	TransportOperator transport_op(D, Linv, Ms_form, psi); 
// 	std::unique_ptr<DiffusionSyntheticAccelerationOperator> dsa_op; 
// 	if (dsa_solver) 
// 		dsa_op = std::make_unique<DiffusionSyntheticAccelerationOperator>(*dsa_solver, Ms_form); 
// 	if (dsa_op) outer_solver->SetPreconditioner(*dsa_op); 
// 	outer_solver->SetOperator(transport_op); 
// 	outer_solver->Mult(phi_source, phi); 

// 	// solve for temperature update 
// 	Mtot.Mult(phi, phi_source); 
// 	phi_source += temp_resid; 
// 	meb_grad_inv.Mult(phi_source, dT); 
// }