#include "trt_linearized.hpp"
#include "transport_op.hpp"

LinearizedTRTOperator::LinearizedTRTOperator(
	const mfem::Array<int> &offsets, 
	const InverseAdvectionOperator &Linv, 
	const mfem::Operator &D, 
	const mfem::Operator &B, 
	const mfem::Operator &Bt, 
	const mfem::Operator &sigma, 
	mfem::IterativeSolver &nonlinear_solver,
	mfem::IterativeSolver &schur_solver, 
	mfem::Solver &meb_grad_inv, 
	const mfem::Solver *dsa_solver)
	: offsets(offsets), Linv(Linv), D(D), B(B), Bt(Bt), sigma(sigma), 
	  nonlinear_solver(nonlinear_solver), schur_solver(schur_solver), 
	  meb_grad_inv(meb_grad_inv), dsa_solver(dsa_solver)
{
	height = width = offsets.Last(); 

	reduced_offsets.SetSize(3); 
	reduced_offsets[0] = 0; 
	reduced_offsets[1] = D.Height(); 
	reduced_offsets[2] = Bt.Height(); 
	reduced_offsets.PartialSum(); 

	reduced_x = std::make_unique<mfem::BlockVector>(reduced_offsets); 
	(*reduced_x) = 0.0; 
	reduced_b = std::make_unique<mfem::BlockVector>(reduced_offsets); 
}

void LinearizedTRTOperator::Mult(const mfem::Vector &x, mfem::Vector &y) const 
{
	const mfem::Vector source_psi(*const_cast<mfem::Vector*>(&x), offsets[0], offsets[1] - offsets[0]); 
	const mfem::Vector source_T(*const_cast<mfem::Vector*>(&x), offsets[1], offsets[2] - offsets[1]); 
	mfem::Vector psi(y, offsets[0], offsets[1] - offsets[0]); 
	mfem::Vector T(y, offsets[1], offsets[2] - offsets[1]); 

	bool use_fixup = Linv.IsFixupOn(); 
	// will return to original state at end of Mult => not violate const-ness of Linv 
	const_cast<InverseAdvectionOperator*>(&Linv)->UseFixup(false); 

	// eliminate psi 
	Linv.Mult(source_psi, psi); 
	D.Mult(psi, reduced_b->GetBlock(0)); 
	reduced_b->GetBlock(1) = source_T;

	NonlinearOperator op(reduced_offsets, Linv, D, B, Bt, sigma, psi); 
	JacobianSolver grad_inv(reduced_offsets, schur_solver, meb_grad_inv, dsa_solver); 
	nonlinear_solver.SetPreconditioner(grad_inv);
	nonlinear_solver.SetOperator(op); 
	nonlinear_solver.Mult(*reduced_b, *reduced_x); 

	const_cast<InverseAdvectionOperator*>(&Linv)->UseFixup(use_fixup); 
	T = reduced_x->GetBlock(1); 
	B.Mult(T, reduced_b->GetBlock(0)); 
	D.MultTranspose(reduced_b->GetBlock(0), psi); 
	add(psi, 1.0, source_psi, psi); 
	Linv.Mult(psi, psi); 

	if (rebalance_solver and (use_fixup or !nonlinear_solver.GetConverged())) {
		D.Mult(psi, reduced_x->GetBlock(0)); 
		sigma.Mult(reduced_x->GetBlock(0), reduced_b->GetBlock(0)); 
		add(source_T, 1.0, reduced_b->GetBlock(0), reduced_b->GetBlock(0)); 
		rebalance_solver->Mult(reduced_b->GetBlock(0), T); 
	}
}

LinearizedTRTOperator::
NonlinearOperator::NonlinearOperator(
	const mfem::Array<int> &offsets, 
	const mfem::Operator &Linv, 
	const mfem::Operator &D, 
	const mfem::Operator &B, 
	const mfem::Operator &Bt, 
	const mfem::Operator &sigma, 
	mfem::Vector &psi)
	: offsets(offsets), Linv(Linv), D(D), B(B), Bt(Bt), sigma(sigma), psi(psi)
{
	height = width = offsets.Last(); 
	tmp.SetSize(offsets[2] - offsets[1]); 
	grad = std::make_unique<mfem::BlockOperator>(offsets); 
	F11 = std::make_unique<mfem::IdentityOperator>(offsets[1] - offsets[0]); 
}

void LinearizedTRTOperator::
NonlinearOperator::Mult(const mfem::Vector &x, mfem::Vector &y) const 
{
	const mfem::Vector phi(*const_cast<mfem::Vector*>(&x), offsets[0], offsets[1] - offsets[0]); 
	const mfem::Vector T(*const_cast<mfem::Vector*>(&x), offsets[1], offsets[2] - offsets[1]); 
	mfem::Vector f_phi(y, offsets[0], offsets[1] - offsets[0]); 
	mfem::Vector f_T(y, offsets[1], offsets[2] - offsets[1]); 

	B.Mult(T, f_phi); 
	D.MultTranspose(f_phi, psi); 
	Linv.Mult(psi, psi); 
	D.Mult(psi, f_phi); 
	add(phi, -1.0, f_phi, f_phi); // phi - f_phi -> f_phi 

	sigma.Mult(phi, f_T); 
	Bt.Mult(T, tmp); 
	add(tmp, -1.0, f_T, f_T); // tmp - f_T -> f_T 
}

mfem::Operator &LinearizedTRTOperator::
NonlinearOperator::GetGradient(const mfem::Vector &x) const 
{
	const mfem::Vector phi(*const_cast<mfem::Vector*>(&x), offsets[0], offsets[1] - offsets[0]); 
	const mfem::Vector T(*const_cast<mfem::Vector*>(&x), offsets[1], offsets[2] - offsets[1]); 

	F12.reset(new F12Operator(D, Linv, B.GetGradient(T), psi)); 

	grad->SetBlock(0,0, F11.get()); 
	grad->SetBlock(0,1, F12.get(), -1.0);

	grad->SetBlock(1,0, const_cast<mfem::Operator*>(&sigma), -1.0); 
	grad->SetBlock(1,1, &Bt.GetGradient(T)); 

	return *grad; 
}

LinearizedTRTOperator::
JacobianSolver::JacobianSolver(
	const mfem::Array<int> &offsets, mfem::IterativeSolver &schur_solver, 
	mfem::Solver &meb_grad_inv, const mfem::Solver *dsa_solver)
	: offsets(offsets), schur_solver(schur_solver), meb_grad_inv(meb_grad_inv), dsa_solver(dsa_solver)
{
	height = width = offsets.Last(); 
	tmp_phi.SetSize(offsets[1] - offsets[0]); 
	tmp_T.SetSize(offsets[2] - offsets[1]); 
}

void LinearizedTRTOperator::
JacobianSolver::SetOperator(const mfem::Operator &op)
{
	block_op = dynamic_cast<const mfem::BlockOperator*>(&op); 
	if (!block_op) MFEM_ABORT("JacobianSolver requires block operator"); 

	meb_grad_inv.SetOperator(block_op->GetBlock(1,1)); 
	auto *ptr = new SchurComplementOperator(
		block_op->GetBlock(0,1), 
		meb_grad_inv, 
		block_op->GetBlock(1,0)); 
	schur_op.reset(ptr); 
	if (dsa_solver) {
		const auto &dB = dynamic_cast<const F12Operator*>(&block_op->GetBlock(0,1))->GetEmissionGradient(); 
		dsa_Ms.reset(new mfem::TripleProductOperator(&dB, &meb_grad_inv, &block_op->GetBlock(1,0), false, false, false)); 
		dsa_op.reset(new DiffusionSyntheticAccelerationOperator(*dsa_solver, *dsa_Ms)); 
		schur_solver.SetPreconditioner(*dsa_op); 
	}
	schur_solver.SetOperator(*schur_op); 
}

void LinearizedTRTOperator::
JacobianSolver::Mult(const mfem::Vector &b, mfem::Vector &x) const 
{
	const mfem::Vector source_phi(*const_cast<mfem::Vector*>(&b), offsets[0], offsets[1] - offsets[0]); 
	const mfem::Vector source_T(*const_cast<mfem::Vector*>(&b), offsets[1], offsets[2] - offsets[1]); 
	mfem::Vector phi(x, offsets[0], offsets[1] - offsets[0]); 
	mfem::Vector T(x, offsets[1], offsets[2] - offsets[1]); 	

	meb_grad_inv.Mult(source_T, tmp_T); 
	block_op->GetBlock(0,1).Mult(tmp_T, tmp_phi); 
	add(source_phi, 1.0, tmp_phi, tmp_phi); 

	schur_solver.Mult(tmp_phi, phi); 

	block_op->GetBlock(1,0).Mult(phi, tmp_T); 
	add(source_T, 1.0, tmp_T, tmp_T); 
	meb_grad_inv.Mult(tmp_T, T); 
}