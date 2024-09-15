#include "trt_ndsa.hpp"
#include "log.hpp"

NonlinearDSATRTOperator::NonlinearDSATRTOperator(
	const mfem::Array<int> &offsets, // [psi, T]
	const mfem::Operator &Linv, // sweep
	const mfem::Operator &D, // discrete to moment 
	const mfem::Operator &G,
	const mfem::Operator &B, // mg emission
	const DenseBlockDiagonalNonlinearForm &Bgr, // gray emission
	const DenseBlockDiagonalNonlinearForm &Bt, // emission with Cv/dt term
	mfem::BilinearForm &Mtot, // M_sigma 
	const MomentDiscretization &lo_disc,
	mfem::Solver &linear_solver, // solves linear transport problem 
	mfem::IterativeSolver &lo_solver,
	const mfem::Solver &meb_solver, 
	mfem::Vector &moments_nu, 
	mfem::Vector &moments_gray
	)
	: offsets(offsets), Linv(Linv), D(D), G(G), B(B), Bgr(Bgr), Bt(Bt), Mtot(Mtot), 
	  lo_disc(lo_disc), linear_solver(linear_solver), lo_solver(lo_solver), meb_solver(meb_solver), 
	  moments_nu(moments_nu), moments_gray(moments_gray)
{
	height = width = offsets.Last();
	lo_offsets.SetSize(3);
	lo_offsets[0] = 0;
	lo_offsets[1] = Bt.Height();
	lo_offsets[2] = Bgr.Height();
	lo_offsets.PartialSum();

	em_source.SetSize(B.Height());
	em_source_gr.SetSize(Bgr.Height());
	phi.SetSize(G.Height());
	phi = 0.0;
	residual_phi.SetSize(G.Height());

	lo_source.SetSize(lo_offsets.Last());
	lo_soln.SetSize(lo_offsets.Last());
}

void NonlinearDSATRTOperator::Mult(const mfem::Vector &x, mfem::Vector &y) const 
{
	mfem::Vector source_psi(const_cast<mfem::Vector&>(x), offsets[0], offsets[1] - offsets[0]);
	mfem::Vector source_T(const_cast<mfem::Vector&>(x), offsets[1], offsets[2] - offsets[1]);

	mfem::Vector psi(y, offsets[0], offsets[1] - offsets[0]);
	mfem::Vector T(y, offsets[1], offsets[2] - offsets[1]);

	B.Mult(T, em_source);
	D.MultTranspose(em_source, psi);
	psi += source_psi;
	Linv.Mult(psi, psi);

	D.Mult(psi, moments_nu);
	G.Mult(moments_nu, moments_gray);

	mfem::tic();
	for (auto *coef : opacities) {
		coef->Project();
	}
	TimingLog.Log("coef project", mfem::toc());

	Mtot = 0.0;
	Mtot.Assemble();

	auto K = std::unique_ptr<mfem::HypreParMatrix>(lo_disc.GetOperator());
	Bgr.Mult(T, em_source_gr);
	K->Mult(moments_gray, residual_phi);
	add(residual_phi, -1.0, em_source_gr, residual_phi);

	mfem::BlockVector block_lo_source(lo_source, 0, lo_offsets);
	mfem::BlockVector block_lo_soln(lo_soln, 0, lo_offsets);
	block_lo_source.GetBlock(0) = source_T;
	block_lo_source.GetBlock(1) = residual_phi;
	block_lo_soln.GetBlock(0) = T;
	block_lo_soln.GetBlock(1) = moments_gray;

	LowOrderOperator op(lo_offsets, *K, Bgr, Bt, Mtot, meb_solver, linear_solver, block_lo_source);
	lo_solver.SetOperator(op);
	mfem::Vector blank;
	lo_solver.Mult(blank, block_lo_soln);

	T = block_lo_soln.GetBlock(0);
	phi = block_lo_soln.GetBlock(1);
}

NonlinearDSATRTOperator::
LowOrderOperator::LowOrderOperator(
	const mfem::Array<int> &offsets, 
	const mfem::HypreParMatrix &K,
	const DenseBlockDiagonalNonlinearForm &B, 
	const DenseBlockDiagonalNonlinearForm &Bt, 
	const mfem::BilinearForm &Mtot, 
	const mfem::Solver &meb_solver, 
	mfem::Solver &lo_solver,
	const mfem::Vector &source)
	: offsets(offsets), K(K), B(B), Bt(Bt), 
	  Mtot(Mtot), meb_solver(meb_solver), lo_solver(lo_solver), source(source), dB_dBt_inv(*Bt.FESpace())
{
	height = width = offsets.Last();
	emission.SetSize(B.Height());
	meb_eval.SetSize(Bt.Height());
	tmp_phi.SetSize(Mtot.Height());
	tmp_T.SetSize(Bt.Height());
}

void NonlinearDSATRTOperator::
LowOrderOperator::Mult(const mfem::Vector &blank, mfem::Vector &x) const 
{
	const mfem::Vector source_T(const_cast<mfem::Vector&>(source), offsets[0], offsets[1] - offsets[0]);
	const mfem::Vector source_phi(const_cast<mfem::Vector&>(source), offsets[1], offsets[2] - offsets[1]);

	mfem::Vector T(x, offsets[0], offsets[1] - offsets[0]);
	mfem::Vector phi(x, offsets[1], offsets[2] - offsets[1]);

	Mtot.Mult(phi, tmp_phi);
	add(tmp_phi, 1.0, source_T, tmp_T);
	meb_solver.Mult(tmp_T, T);

	B.Mult(T, emission);
	Bt.Mult(T, meb_eval);
	add(source_T, -1.0, meb_eval, tmp_T);
	add(emission, 1.0, source_phi, tmp_phi);

	const auto &dB = B.GetGradient(T);
	auto &dBt_inv = Bt.GetGradient(T);
	dBt_inv.Invert();
	::Mult(dB, dBt_inv, dB_dBt_inv);
	auto lin_emission = std::unique_ptr<mfem::SparseMatrix>(::Mult(dB_dBt_inv, Mtot.SpMat())); 

	mfem::HypreParMatrix schur(K);
	mfem::SparseMatrix diag;
	schur.GetDiag(diag);
	diag.Add(-1.0, *lin_emission);

	dB_dBt_inv.AddMult(tmp_T, tmp_phi);

	lo_solver.SetOperator(schur);
	lo_solver.Mult(tmp_phi, phi);
}
