#include "smm_op.hpp"
#include "p1diffusion.hpp"
#include "linalg.hpp"

LDGDiffusionDiscretization::LDGDiffusionDiscretization(mfem::ParFiniteElementSpace &_fes, mfem::ParFiniteElementSpace &_vfes, 
	mfem::Coefficient &_total, mfem::Coefficient &_absorption, double _alpha)
	: fes(_fes), vfes(_vfes), total(_total), absorption(_absorption), alpha(_alpha)
{
	offsets.SetSize(3); 
	offsets[0] = 0; 
	offsets[1] = vfes.GetVSize(); 
	offsets[2] = fes.GetVSize(); 
	offsets.PartialSum(); 
	const auto dim = fes.GetMesh()->Dimension(); 

	mfem::ParBilinearForm Mtform(&vfes);
	mfem::ProductCoefficient total3(3.0, total); 
	Mtform.AddDomainIntegrator(new mfem::VectorMassIntegrator(total3)); 
	Mtform.AddBdrFaceIntegrator(new DGVectorJumpJumpIntegrator(1./2/alpha)); 
	Mtform.Assemble(); 
	Mtform.Finalize();  
	auto Mt = HypreParMatrixPtr(Mtform.ParallelAssemble()); 
	iMt = HypreParMatrixPtr(ElementByElementBlockInverse(vfes, *Mt)); 

	mfem::ParBilinearForm Maform(&fes); 
	mfem::ConstantCoefficient alpha_c(alpha/2); 
	Maform.AddDomainIntegrator(new mfem::MassIntegrator(absorption)); 
	Maform.AddInteriorFaceIntegrator(new PenaltyIntegrator(alpha/2, false)); 
	Maform.AddBdrFaceIntegrator(new mfem::BoundaryMassIntegrator(alpha_c)); 
	Maform.Assemble(); 
	Maform.Finalize(); 
	Ma = HypreParMatrixPtr(Maform.ParallelAssemble());

	mfem::ParMixedBilinearForm Dform(&vfes, &fes); 
	mfem::ConstantCoefficient neg_one(-1.0); 
	Dform.AddDomainIntegrator(new mfem::TransposeIntegrator(new mfem::GradientIntegrator(neg_one))); 
	mfem::Vector ldg_beta(dim); 
	for (auto d=0; d<dim; d++) { ldg_beta(d) = d+1; }
	Dform.AddInteriorFaceIntegrator(new mfem::LDGTraceIntegrator(&ldg_beta)); 
	Dform.AddBdrFaceIntegrator(new DGJumpAverageIntegrator(0.5)); 
	Dform.Assemble(); 
	Dform.Finalize(); 
	D = HypreParMatrixPtr(Dform.ParallelAssemble()); 
	DT = HypreParMatrixPtr(D->Transpose()); 

	auto DiMt = HypreParMatrixPtr(mfem::ParMult(D.get(), iMt.get(), true)); 
	auto DiMtDT = HypreParMatrixPtr(mfem::ParMult(DiMt.get(), DT.get(), true)); 
	S = HypreParMatrixPtr(mfem::ParAdd(DiMtDT.get(), Ma.get())); 
}

void LDGDiffusionDiscretization::EliminateRHS(const mfem::Vector &g, mfem::Vector &f) const 
{
	tmp_elim_vec.SetSize(iMt->Height());
	iMt->Mult(g, tmp_elim_vec); 
	D->Mult(-1.0, tmp_elim_vec, 1.0, f);  
}

void LDGDiffusionDiscretization::BackSolve(const mfem::Vector &g, const mfem::Vector &phi, mfem::Vector &J) const
{
	tmp_elim_vec.SetSize(DT->Height()); 
	DT->Mult(phi, tmp_elim_vec); 
	tmp_elim_vec += g; 
	iMt->Mult(tmp_elim_vec, J); 
}

void InverseLDGDiffusionOperator::Mult(const mfem::Vector &b, mfem::Vector &x) const 
{
	const auto &offsets = disc.GetOffsets(); 
	mfem::BlockVector bb(b.GetData(), offsets); 
	mfem::BlockVector bx(x.GetData(), offsets);

	const auto &S = disc.SchurComplement(); 
	phi_source = bb.GetBlock(1); 
	disc.EliminateRHS(bb.GetBlock(0), phi_source); 
	Sinv.Mult(phi_source, bx.GetBlock(1)); 
	disc.BackSolve(bb.GetBlock(0), bx.GetBlock(1), bx.GetBlock(0)); 
}

LDGSMMSourceOperator::LDGSMMSourceOperator(mfem::ParFiniteElementSpace &_fes, mfem::ParFiniteElementSpace &_vfes, 
	const AngularQuadrature &_quad, const TransportVectorExtents &_psi_ext, PhaseSpaceCoefficient &source_coef, 
	PhaseSpaceCoefficient &inflow_coef, double _alpha)
	: fes(_fes), vfes(_vfes), quad(_quad), psi_ext(_psi_ext), alpha(_alpha)
{
	offsets.SetSize(3); 
	offsets[0] = 0; 
	offsets[1] = vfes.GetVSize(); 
	offsets[2] = fes.GetVSize(); 
	offsets.PartialSum(); 

	width = TotalExtent(psi_ext); 
	height = offsets.Last(); 

	const auto dim = fes.GetMesh()->Dimension(); 
	Q0.SetSize(fes.GetVSize()); 
	Q0 = 0.0; 
	Q1.SetSize(vfes.GetVSize()); 
	Q1 = 0.0; 

	for (auto a=0; a<quad.Size(); a++) {
		const auto &Omega = quad.GetOmega(a); 
		source_coef.SetState(Omega); 
		mfem::LinearForm fform(&fes); 
		fform.AddDomainIntegrator(new mfem::DomainLFIntegrator(source_coef)); 
		fform.Assemble(); 

		mfem::VectorConstantCoefficient Omega_coef(Omega); 
		mfem::ScalarVectorProductCoefficient Omega_source(source_coef, Omega_coef);
		mfem::LinearForm gform(&vfes); 
		gform.AddDomainIntegrator(new mfem::VectorDomainLFIntegrator(Omega_source)); 
		gform.Assemble();  

		Q0.Add(quad.GetWeight(a), fform); 
		Q1.Add(quad.GetWeight(a), gform); 
	}

	InflowPartialCurrentCoefficient Jin(inflow_coef, quad); 
	mfem::LinearForm fform(&fes); 
	fform.AddBdrFaceIntegrator(new mfem::BoundaryLFIntegrator(Jin, 2, 1)); 
	fform.Assemble(); 

	mfem::LinearForm gform(&vfes); 
	gform.AddBdrFaceIntegrator(new BoundaryNormalFaceLFIntegrator(Jin, 2, 1)); 
	gform.Assemble(); 

	Q0.Add(-1.0, fform); 
	Q1.Add(1./alpha/3, gform); 
	Q1 *= 3.0; 
}

void LDGSMMSourceOperator::Mult(const mfem::Vector &psi, mfem::Vector &source) const 
{
	mfem::BlockVector bv(source.GetData(), offsets); 
	TransportVectorView psi_view(psi.GetData(), psi_ext); 
	SMMCorrectionTensorCoefficient T(fes, quad, psi_view); 
	SMMBdrCorrectionFactorCoefficient beta(fes, quad, psi_view, alpha); 	

	mfem::ParLinearForm fform(&fes, bv.GetBlock(1).GetData()); 
	mfem::ProductCoefficient bdr_coef_f(-0.5, beta); 
	fform.AddBdrFaceIntegrator(new mfem::BoundaryLFIntegrator(bdr_coef_f, 2, 1)); 
	fform.Assemble(); 

	mfem::ParLinearForm gform(&vfes, bv.GetBlock(0).GetData()); 
	gform.AddDomainIntegrator(new WeakTensorDivergenceLFIntegrator(T)); 
	gform.AddInteriorFaceIntegrator(new VectorJumpTensorAverageLFIntegrator(T)); 
	gform.AddBdrFaceIntegrator(new VectorJumpTensorAverageLFIntegrator(T)); 
	mfem::ProductCoefficient bdr_coef_g(1./2/alpha/3, beta);
	gform.AddBdrFaceIntegrator(new BoundaryNormalFaceLFIntegrator(bdr_coef_g, 2, 1)); 
	gform.Assemble();
	gform *= 3.0;  

	fform += Q0; 
	gform += Q1; 
}