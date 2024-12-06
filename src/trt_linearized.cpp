#include "trt_linearized.hpp"
#include "transport_op.hpp"
#include "constants.hpp"
#include "log.hpp"
#include "moment_discretization.hpp"
#include "planck.hpp"

namespace experimental
{

NewtonTRTOperator::NewtonTRTOperator(
	const mfem::Array<int> &offsets, 
	const InverseAdvectionOperator &Linv, 
	const mfem::Operator &D, 
	const mfem::Operator &B, 
	const mfem::Operator &Bt, 
	const mfem::Operator &sigma, 
	mfem::IterativeSolver &nonlinear_solver,
	mfem::IterativeSolver &schur_solver, 
	mfem::Solver &meb_grad_inv)
	: offsets(offsets), Linv(Linv), D(D), B(B), Bt(Bt), sigma(sigma), 
	  nonlinear_solver(nonlinear_solver), schur_solver(schur_solver), 
	  meb_grad_inv(meb_grad_inv)
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

void NewtonTRTOperator::Mult(const mfem::Vector &x, mfem::Vector &y) const 
{
	const mfem::Vector source_psi(*const_cast<mfem::Vector*>(&x), offsets[0], offsets[1] - offsets[0]); 
	const mfem::Vector source_T(*const_cast<mfem::Vector*>(&x), offsets[1], offsets[2] - offsets[1]); 
	mfem::Vector psi(y, offsets[0], offsets[1] - offsets[0]); 
	mfem::Vector T(y, offsets[1], offsets[2] - offsets[1]); 

	// eliminate psi 
	Linv.Mult(source_psi, psi); 
	D.Mult(psi, reduced_b->GetBlock(0)); 
	reduced_b->GetBlock(1) = source_T;

	// will return to original state at end of Mult => not violate const-ness of Linv 
	const bool use_fixup = Linv.IsFixupOn(); 
	const_cast<InverseAdvectionOperator*>(&Linv)->UseFixup(false); 

	NonlinearOperator op(reduced_offsets, Linv, D, B, Bt, sigma, psi); 
	op.SetSource(*reduced_b); 
	JacobianSolver grad_inv(reduced_offsets, schur_solver, meb_grad_inv, dsa_disc, dsa_solver); 
	nonlinear_solver.SetOperator(op); 
	nonlinear_solver.SetPreconditioner(grad_inv);
	auto *kinsol = dynamic_cast<mfem::KINSolver*>(&nonlinear_solver); 
	D.Mult(psi, reduced_x->GetBlock(0)); 
	reduced_x->GetBlock(1) = T; 
	mfem::Vector blank; 
	if (kinsol) {
		kinsol->SetMaxSetupCalls(1); 
		mfem::BlockVector xscale(reduced_offsets), fscale(reduced_offsets); 
		// xscale.GetBlock(0) = 1.0/reduced_x->GetBlock(0).Normlinf(); 
		// xscale.GetBlock(1) = 1.0/T.Normlinf(); 
		op.Mult(*reduced_x, fscale); 
		fscale = 1.0/fscale.Normlinf(); 
		xscale.GetBlock(0) = 1.0/reduced_x->GetBlock(0).Normlinf(); 
		xscale.GetBlock(1) = 1.0/reduced_x->GetBlock(1).Normlinf(); 
		auto flag = KINSetFuncNormTol(kinsol->GetMem(), 1e-4);
		MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINSetFuncNormTol()");
		kinsol->Mult(*reduced_x, xscale, fscale); 
		// kinsol->Mult(blank, *reduced_x); 
	} else {
		nonlinear_solver.Mult(blank, *reduced_x); 				
	}

	T = reduced_x->GetBlock(1); 
	B.Mult(T, reduced_b->GetBlock(0)); 
	D.MultTranspose(reduced_b->GetBlock(0), psi); 
	add(psi, 1.0, source_psi, psi); 
	// re-enable fixup if enabled on entry 
	const_cast<InverseAdvectionOperator*>(&Linv)->UseFixup(use_fixup); 
	Linv.Mult(psi, psi); 

	// if (rebalance_solver and (use_fixup or !nonlinear_solver.GetConverged())) {
	// if (rebalance_solver) {
	// 	D.Mult(psi, reduced_x->GetBlock(0)); 
	// 	sigma.Mult(reduced_x->GetBlock(0), reduced_b->GetBlock(0)); 
	// 	add(source_T, 1.0, reduced_b->GetBlock(0), reduced_b->GetBlock(0)); 
	// 	rebalance_solver->Mult(reduced_b->GetBlock(0), T); 
	// }
}

NewtonTRTOperator::
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

void NewtonTRTOperator::
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
	if (source) y -= *source; 
}

mfem::Operator &NewtonTRTOperator::
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

NewtonTRTOperator::
JacobianSolver::JacobianSolver(
	const mfem::Array<int> &offsets, mfem::IterativeSolver &schur_solver, 
	mfem::Solver &meb_grad_inv, const MomentDiscretization *dsa_disc, mfem::Solver *dsa_solver)
	: offsets(offsets), schur_solver(schur_solver), meb_grad_inv(meb_grad_inv), 
	  dsa_disc(dsa_disc), dsa_solver(dsa_solver)
{
	height = width = offsets.Last(); 
	tmp_phi.SetSize(offsets[1] - offsets[0]); 
	tmp_T.SetSize(offsets[2] - offsets[1]); 
}

void NewtonTRTOperator::
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
	// if (dsa_solver) {
	// 	const auto &dB = dynamic_cast<const F12Operator*>(&block_op->GetBlock(0,1))->GetEmissionGradient(); 
	// 	dsa_Ms.reset(new mfem::TripleProductOperator(&dB, &meb_grad_inv, &block_op->GetBlock(1,0), false, false, false)); 
	// 	dsa_op.reset(dsa_disc->GetOperator());
	// 	dsa_solver->SetOperator(*dsa_op);
	// 	dsa_prec.reset(new DiffusionSyntheticAccelerationOperator(*dsa_solver, *dsa_Ms)); 
	// 	schur_solver.SetPreconditioner(*dsa_prec); 
	// }
	schur_solver.SetOperator(*schur_op); 
}

void NewtonTRTOperator::
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

} // end namespace experimental 

LinearizedTRTOperator::LinearizedTRTOperator(
	const mfem::Array<int> &offsets, 
	InverseAdvectionOperator &Linv, 
	const mfem::Operator &D, 
	const mfem::Operator &B, 
	const mfem::Operator &Bt, 
	const mfem::Operator &sigma, 
	mfem::IterativeSolver &schur_solver, 
	mfem::Solver &meb_grad_inv)
	: offsets(offsets), Linv(Linv), D(D), B(B), Bt(Bt), sigma(sigma), 
	  schur_solver(schur_solver), meb_grad_inv(meb_grad_inv)
{
	height = width = offsets.Last(); 
	temp_resid.SetSize(Bt.Height()); 
	dT.SetSize(Bt.Height()); 
	em_source.SetSize(D.Height()); 
	phi_source.SetSize(D.Height()); 
	abs_source.SetSize(sigma.Height());
	phi.SetSize(D.Height()); 
	t1.SetSize(Bt.Height()); 
}

void LinearizedTRTOperator::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
	const mfem::Vector source_psi(*const_cast<mfem::Vector*>(&x), offsets[0], offsets[1] - offsets[0]); 
	const mfem::Vector source_T(*const_cast<mfem::Vector*>(&x), offsets[1], offsets[2] - offsets[1]); 
	mfem::Vector psi(y, offsets[0], offsets[1] - offsets[0]); 
	mfem::Vector T(y, offsets[1], offsets[2] - offsets[1]); 

	const bool use_fixup = Linv.IsFixupOn(); 
	// disable fixup for newton operations 
	const_cast<InverseAdvectionOperator*>(&Linv)->UseFixup(false); 

	for (auto *ptr : opacities) {
		ptr->Project();
	}

	// initial guess for schur solve 
	D.Mult(psi, phi); 

	// --- form RHS --- 
	Bt.Mult(T, temp_resid); 
	add(source_T, -1.0, temp_resid, temp_resid); // T0 - temp_resid -> temp_resid 

	// invert (cv/dt + 4a sigma T^3)
	const auto &Bt_grad = Bt.GetGradient(T); 
	meb_grad_inv.SetOperator(Bt_grad); 
	// ( 4a sigma T^3 u, v)
	const auto &dplanck = B.GetGradient(T); 

	// form transport residual 
	B.Mult(T, em_source); 
	// eliminate down to phi
	meb_grad_inv.Mult(temp_resid, t1); 
	dplanck.Mult(t1, phi_source); 
	em_source += phi_source; 
	D.MultTranspose(em_source, psi); 
	psi += source_psi; 
	Linv.Mult(psi, psi);
	D.Mult(psi, phi_source); 

	// apply I - D Linv dB (dBt)^{-1} Msigma 
	mfem::TripleProductOperator Ms_form(&dplanck, &meb_grad_inv, &sigma, false, false, false); 
	TransportOperator transport_op(D, Linv, Ms_form, psi); 
	std::unique_ptr<DiffusionSyntheticAccelerationOperator> dsa_op; 
	if (dsa_solver) {
		const auto &solver = dsa_solver->GetSolver();
		dsa_op = std::make_unique<DiffusionSyntheticAccelerationOperator>(solver, Ms_form); 
		schur_solver.SetPreconditioner(*dsa_op);
	}
	schur_solver.SetOperator(transport_op); 
	schur_solver.Mult(phi_source, phi); 

	// sweep to recover psi 
	Ms_form.Mult(phi, phi_source); 
	em_source += phi_source;
	D.MultTranspose(em_source, psi); 
	psi += source_psi; 
	// re-enable fixup if on at entry 
	// returns Linv to original state => doesn't violate const contract 
	const_cast<InverseAdvectionOperator*>(&Linv)->UseFixup(use_fixup); 
	Linv.Mult(psi, psi);

	// solve for temperature update 
	if (rebalance_solver) {
		D.Mult(psi, phi); 
		sigma.Mult(phi, abs_source); 
		abs_source += source_T; 
		rebalance_solver->Mult(abs_source, T); 
	} else {
		sigma.Mult(phi, abs_source); 
		abs_source += temp_resid; 
		meb_grad_inv.Mult(abs_source, dT); 
		for (int i=0; i<T.Size(); i++) {
			double Tnew = T(i) + dT(i); 
			if (Tnew < 0.0) {
				EventLog.Register("under relax"); 
				T(i) = T(i) + 0.05*dT(i); 
			} else {
				T(i) = Tnew; 
			}
		}
	}
}

double LinearizedPseudoAbsorptionCoefficient::Eval(mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) 
{
	const auto temperature = T.Eval(trans, ip);
	const auto heat_capacity = Cvdt.Eval(trans, ip);
	const auto opacity = sigma.Eval(trans, ip);
	const double dB = 4.0 * opacity * constants::StefanBoltzmann * pow(temperature,3);
	const double nu = dB / (heat_capacity + dB);
	return opacity * (1.0 - nu);
}

InexactNewtonTRTOperator::InexactNewtonTRTOperator(
	const mfem::Array<int> &offsets, // [psi, T]
	const InverseAdvectionOperator &Linv, // sweep
	const mfem::Operator &D, // discrete to moment 
	const mfem::Operator &B, // emission
	const mfem::Operator &Bt, // emission with Cv/dt term
	const mfem::Operator &sigma, // M_sigma 
	const mfem::Solver &meb_solver, 
	mfem::Solver &lin_meb_solver)
	: offsets(offsets), Linv(Linv), D(D), B(B), Bt(Bt), sigma(sigma), 
	  meb_solver(meb_solver), lin_meb_solver(lin_meb_solver)
{
	height = width = offsets.Last();

	phi.SetSize(D.Height());
	phi2.SetSize(D.Height());
	abs_source.SetSize(sigma.Height());
	temp_resid.SetSize(Bt.Height());
	t1.SetSize(Bt.Height());
	t2.SetSize(B.Height());
	em_source.SetSize(B.Height());
}

void InexactNewtonTRTOperator::Mult(const mfem::Vector &x, mfem::Vector &y) const 
{
	const mfem::Vector source_psi(const_cast<mfem::Vector&>(x), offsets[0] , offsets[1] - offsets[0]);
	const mfem::Vector source_T(const_cast<mfem::Vector&>(x), offsets[1] , offsets[2] - offsets[1]);

	mfem::Vector psi(y, offsets[0], offsets[1] - offsets[0]);
	mfem::Vector T(y, offsets[1], offsets[2] - offsets[1]);

	for (auto *ptr : opacities) {
		ptr->Project();
	}

	D.Mult(psi, phi);
	sigma.Mult(phi, abs_source);
	abs_source += source_T;

	const auto &dBt = Bt.GetGradient(T);
	lin_meb_solver.SetOperator(dBt);
	const auto &dB = B.GetGradient(T);

	Bt.Mult(T, temp_resid);
	add(abs_source, -1.0, temp_resid, temp_resid);
	lin_meb_solver.Mult(temp_resid, t1);
	dB.Mult(t1, t2);
	B.Mult(T, em_source);
	em_source += t2;
	D.MultTranspose(em_source, psi);
	psi += source_psi;
	Linv.Mult(psi, psi);

	if (dsa_solver) {
		D.Mult(psi, phi2); 
		phi2 -= phi;

		sigma.Mult(phi2, abs_source);
		lin_meb_solver.Mult(abs_source, t1);
		dB.Mult(t1, t2);

		const auto &solver = dsa_solver->GetSolver();
		solver.Mult(t2, phi);
		D.AddMultTranspose(phi, psi);		
	}

	D.Mult(psi, phi);
	sigma.Mult(phi, abs_source);
	abs_source += source_T;
	meb_solver.Mult(abs_source, T);
}
