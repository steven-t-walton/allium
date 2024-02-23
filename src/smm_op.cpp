#include "smm_op.hpp"
#include "p1diffusion.hpp"
#include "linalg.hpp"
#include "mip.hpp"

void BlockDiffusionDiscretization::EliminateRHS(const mfem::Vector &g, mfem::Vector &f) const 
{
	tmp_elim_vec.SetSize(iMt->Height());
	iMt->Mult(g, tmp_elim_vec); 
	D->Mult(-1.0, tmp_elim_vec, 1.0, f);  
}

void BlockDiffusionDiscretization::BackSolve(const mfem::Vector &g, const mfem::Vector &phi, mfem::Vector &J) const
{
	tmp_elim_vec.SetSize(DT->Height()); 
	DT->Mult(phi, tmp_elim_vec); 
	tmp_elim_vec += g; 
	iMt->Mult(tmp_elim_vec, J); 
}

LDGDiffusionDiscretization::LDGDiffusionDiscretization(mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &vfes, 
	mfem::Coefficient &total, mfem::Coefficient &absorption, double alpha, const mfem::Vector &beta, 
	bool scale_ldg_stabilization, int reflect_bdr_attr)
{
	offsets.SetSize(3); 
	offsets[0] = 0; 
	offsets[1] = vfes.GetVSize(); 
	offsets[2] = fes.GetVSize(); 
	offsets.PartialSum(); 

	const auto &mesh = *fes.GetParMesh(); 
	const auto &mesh_bdr_attributes = mesh.bdr_attributes; 
	mfem::Array<int> marshak_bdr_attrs(mesh_bdr_attributes.Max()), reflect_bdr_attrs(mesh_bdr_attributes.Max()); 
	marshak_bdr_attrs = 1; 
	reflect_bdr_attrs = 0; 
	if (reflect_bdr_attr > 0) {
		marshak_bdr_attrs[reflect_bdr_attr-1] = 0; 
		reflect_bdr_attrs[reflect_bdr_attr-1] = 1; 
	}
	const auto dim = mesh.Dimension(); 

	mfem::ParBilinearForm Mtform(&vfes);
	mfem::ProductCoefficient total3(3.0, total); 
	Mtform.AddDomainIntegrator(new mfem::VectorMassIntegrator(total3)); 
	Mtform.AddBdrFaceIntegrator(new DGVectorJumpJumpIntegrator(1./2/alpha), marshak_bdr_attrs); 
	Mtform.AddBdrFaceIntegrator(new DGVectorJumpJumpIntegrator(1.0/alpha), reflect_bdr_attrs); 
	Mtform.Assemble(); 
	Mtform.Finalize();  
	auto Mt = HypreParMatrixPtr(Mtform.ParallelAssemble()); 
	iMt = HypreParMatrixPtr(ElementByElementBlockInverse(vfes, *Mt)); 

	mfem::ParBilinearForm Maform(&fes); 
	mfem::ConstantCoefficient alpha_c(alpha/2); 
	Maform.AddDomainIntegrator(new mfem::MassIntegrator(absorption)); 
	Maform.AddInteriorFaceIntegrator(new PenaltyIntegrator(alpha/2, false)); 
	Maform.AddBdrFaceIntegrator(new mfem::BoundaryMassIntegrator(alpha_c), marshak_bdr_attrs); 
	Maform.Assemble(); 
	Maform.Finalize(); 
	Ma = HypreParMatrixPtr(Maform.ParallelAssemble());

	mfem::ParMixedBilinearForm Dform(&vfes, &fes); 
	mfem::ConstantCoefficient neg_one(-1.0); 
	Dform.AddDomainIntegrator(new mfem::TransposeIntegrator(new mfem::GradientIntegrator(neg_one))); 
	mfem::RatioCoefficient diffco(1.0/3, total); 
	if (scale_ldg_stabilization) {
		double kappa = pow(fes.GetOrder(0)+1, 2); 
		Dform.AddInteriorFaceIntegrator(new LDGTraceIntegrator(diffco, beta, kappa, alpha/2)); 
	} else {
		Dform.AddInteriorFaceIntegrator(new mfem::LDGTraceIntegrator(&beta)); 		
	}
	Dform.AddBdrFaceIntegrator(new DGJumpAverageIntegrator(0.5), marshak_bdr_attrs); 
	Dform.Assemble(); 
	Dform.Finalize(); 
	D = HypreParMatrixPtr(Dform.ParallelAssemble()); 
	DT = HypreParMatrixPtr(D->Transpose()); 

	auto DiMt = HypreParMatrixPtr(mfem::ParMult(D.get(), iMt.get(), true)); 
	auto DiMtDT = HypreParMatrixPtr(mfem::ParMult(DiMt.get(), DT.get(), true)); 
	S = HypreParMatrixPtr(mfem::ParAdd(DiMtDT.get(), Ma.get())); 
}

IPDiffusionDiscretization::IPDiffusionDiscretization(mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &vfes, 
	mfem::Coefficient &total, mfem::Coefficient &absorption, double alpha, double kappa, 
	bool mip, bool scale_ip_stabilization, int reflect_bdr_attr)
{
	offsets.SetSize(3); 
	offsets[0] = 0; 
	offsets[1] = vfes.GetVSize(); 
	offsets[2] = fes.GetVSize(); 
	offsets.PartialSum(); 

	if (kappa < 0) {
		kappa *= -1.0 * pow(fes.GetOrder(0)+1, 2); 
	}

	const auto &mesh = *fes.GetParMesh(); 
	const auto &mesh_bdr_attributes = mesh.bdr_attributes; 
	mfem::Array<int> marshak_bdr_attrs(mesh_bdr_attributes.Max()), reflect_bdr_attrs(mesh_bdr_attributes.Max()); 
	marshak_bdr_attrs = 1; 
	reflect_bdr_attrs = 0; 
	if (reflect_bdr_attr > 0) {
		marshak_bdr_attrs[reflect_bdr_attr-1] = 0; 
		reflect_bdr_attrs[reflect_bdr_attr-1] = 1; 
	}
	const auto dim = mesh.Dimension(); 

	mfem::ParBilinearForm Mtform(&vfes);
	mfem::ProductCoefficient total3(3.0, total); 
	Mtform.AddDomainIntegrator(new mfem::VectorMassIntegrator(total3)); 
	Mtform.AddBdrFaceIntegrator(new DGVectorJumpJumpIntegrator(1./2/alpha), marshak_bdr_attrs); 
	Mtform.AddBdrFaceIntegrator(new DGVectorJumpJumpIntegrator(1.0/alpha), reflect_bdr_attrs); 
	Mtform.Assemble(); 
	Mtform.Finalize();  
	auto Mt = HypreParMatrixPtr(Mtform.ParallelAssemble()); 
	iMt = HypreParMatrixPtr(ElementByElementBlockInverse(vfes, *Mt)); 

	mfem::ParBilinearForm Maform(&fes); 
	mfem::ConstantCoefficient alpha_c(alpha/2); 
	mfem::RatioCoefficient diffco(1./3, total); 
	Maform.AddDomainIntegrator(new mfem::MassIntegrator(absorption)); 
	mfem::Coefficient *coef = scale_ip_stabilization ? &diffco : nullptr; 
	double limit = mip ? alpha/2 : 0.0; 
	Maform.AddInteriorFaceIntegrator(new PenaltyIntegrator(kappa, limit, coef)); 		
	Maform.AddBdrFaceIntegrator(new mfem::BoundaryMassIntegrator(alpha_c), marshak_bdr_attrs); 
	Maform.Assemble(); 
	Maform.Finalize(); 
	Ma = HypreParMatrixPtr(Maform.ParallelAssemble());

	mfem::ParMixedBilinearForm Dform(&vfes, &fes); 
	mfem::ConstantCoefficient neg_one(-1.0); 
	Dform.AddDomainIntegrator(new mfem::TransposeIntegrator(new mfem::GradientIntegrator(neg_one))); 
	Dform.AddInteriorFaceIntegrator(new DGJumpAverageIntegrator); 
	Dform.AddBdrFaceIntegrator(new DGJumpAverageIntegrator(0.5), marshak_bdr_attrs); 
	Dform.Assemble(); 
	Dform.Finalize(); 
	D = HypreParMatrixPtr(Dform.ParallelAssemble()); 
	DT = HypreParMatrixPtr(D->Transpose()); 

	auto DiMt = HypreParMatrixPtr(mfem::ParMult(D.get(), iMt.get(), true)); 
	auto DiMtDT = HypreParMatrixPtr(mfem::ParMult(DiMt.get(), DT.get(), true)); 
	S = HypreParMatrixPtr(mfem::ParAdd(DiMtDT.get(), Ma.get())); 
}

void InverseBlockDiffusionOperator::Mult(const mfem::Vector &b, mfem::Vector &x) const 
{
	const auto &offsets = disc.GetOffsets(); 
	mfem::BlockVector bb(b.GetData(), offsets); 
	mfem::BlockVector bx(x.GetData(), offsets);

	phi_source = bb.GetBlock(1); 
	disc.EliminateRHS(bb.GetBlock(0), phi_source); 
	Sinv.Mult(phi_source, bx.GetBlock(1)); 
	disc.BackSolve(bb.GetBlock(0), bx.GetBlock(1), bx.GetBlock(0)); 
}

BlockDiffusionSMMSourceOperator::BlockDiffusionSMMSourceOperator(
	mfem::ParFiniteElementSpace &_fes, mfem::ParFiniteElementSpace &_vfes, 
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
	// fform.AddBdrFaceIntegrator(new mfem::BoundaryLFIntegrator(Jin, 2, 1)); 
	fform.AddBdrFaceIntegrator(new ProjectedCoefBoundaryLFIntegrator(Jin, *fes.FEColl(), 2, 1)); 
	fform.Assemble(); 

	mfem::LinearForm gform(&vfes); 
	// gform.AddBdrFaceIntegrator(new BoundaryNormalFaceLFIntegrator(Jin, 2, 1)); 
	gform.AddBdrFaceIntegrator(new ProjectedCoefBoundaryNormalLFIntegrator(Jin, *fes.FEColl(), 2, 1)); 
	gform.Assemble(); 

	Q0.Add(-1.0, fform); 
	Q1.Add(1./alpha/3, gform); 
	Q1 *= 3.0; 
}

void BlockDiffusionSMMSourceOperator::Mult(const mfem::Vector &psi, mfem::Vector &source) const 
{
	mfem::BlockVector bv(source.GetData(), offsets); 
	ConstTransportVectorView psi_view(psi.GetData(), psi_ext); 
	SMMCorrectionTensorCoefficient T(fes, quad, psi_view); 
	SMMBdrCorrectionFactorCoefficient beta(fes, quad, psi_view, alpha); 	

	mfem::ParLinearForm fform(&fes, bv.GetBlock(1).GetData()); 
	mfem::ProductCoefficient bdr_coef_f(-0.5, beta); 
	// fform.AddBdrFaceIntegrator(new mfem::BoundaryLFIntegrator(bdr_coef_f, 2, 1)); 
	fform.AddBdrFaceIntegrator(new ProjectedCoefBoundaryLFIntegrator(bdr_coef_f, *fes.FEColl(), 2, 1)); 
	fform.Assemble(); 

	mfem::ParLinearForm gform(&vfes, bv.GetBlock(0).GetData()); 
	gform.AddDomainIntegrator(new WeakTensorDivergenceLFIntegrator(T)); 
	gform.AddInteriorFaceIntegrator(new VectorJumpTensorAverageLFIntegrator(T)); 
	gform.AddBdrFaceIntegrator(new VectorJumpTensorAverageLFIntegrator(T)); 
	mfem::ProductCoefficient bdr_coef_g(1./2/alpha/3, beta);
	// gform.AddBdrFaceIntegrator(new BoundaryNormalFaceLFIntegrator(bdr_coef_g, 2, 1)); 
	gform.AddBdrFaceIntegrator(new ProjectedCoefBoundaryNormalLFIntegrator(bdr_coef_g, *fes.FEColl(), 2, 1)); 
	gform.Assemble();
	gform *= 3.0;  

	fform += Q0; 
	gform += Q1; 
}