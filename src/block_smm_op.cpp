#include "block_smm_op.hpp"
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

BlockLDGDiffusionDiscretization::BlockLDGDiffusionDiscretization(mfem::ParFiniteElementSpace &_fes, mfem::ParFiniteElementSpace &_vfes, 
	mfem::Coefficient &_total, mfem::Coefficient &_absorption, double _alpha, const mfem::Vector &_beta, 
	bool _scale_ldg_stabilization, int _reflect_bdr_attr, DiffusionBoundaryConditionType _Jbcs)
	: fes(_fes), vfes(_vfes), total(_total), absorption(_absorption), alpha(_alpha), beta(_beta), 
	  scale_ldg_stabilization(_scale_ldg_stabilization), reflect_bdr_attr(_reflect_bdr_attr), Jbcs(_Jbcs)
{
	offsets.SetSize(3); 
	offsets[0] = 0; 
	offsets[1] = vfes.GetVSize(); 
	offsets[2] = fes.GetVSize(); 
	offsets.PartialSum(); 

	const auto &mesh = *fes.GetParMesh(); 
	const auto &mesh_bdr_attributes = mesh.bdr_attributes; 
	marshak_bdr_attrs.SetSize(mesh_bdr_attributes.Max()); 
	reflect_bdr_attrs.SetSize(mesh_bdr_attributes.Max()); 
	marshak_bdr_attrs = 1; 
	reflect_bdr_attrs = 0; 
	if (reflect_bdr_attr > 0) {
		marshak_bdr_attrs[reflect_bdr_attr-1] = 0; 
		reflect_bdr_attrs[reflect_bdr_attr-1] = 1; 
	}
	const auto dim = mesh.Dimension(); 

	const bool is_half_range = Jbcs == DiffusionBoundaryConditionType::HALF_RANGE; 
	const bool is_half_range_ref = Jbcs == DiffusionBoundaryConditionType::HALF_RANGE_REFLECT; 

	mfem::ParBilinearForm Mtform(&vfes);
	mfem::ProductCoefficient total3(3.0, total); 
	Mtform.AddDomainIntegrator(new mfem::VectorMassIntegrator(total3)); 
	if (is_half_range or is_half_range_ref) {
		Mtform.AddBdrFaceIntegrator(new DGVectorJumpJumpIntegrator(1.0/alpha), reflect_bdr_attrs); 		
	}
	if (is_half_range) {
		Mtform.AddBdrFaceIntegrator(new DGVectorJumpJumpIntegrator(1./2/alpha), marshak_bdr_attrs); 
	}
	Mtform.Assemble(); 
	Mtform.Finalize();  
	auto Mt = HypreParMatrixPtr(Mtform.ParallelAssemble()); 
	iMt = HypreParMatrixPtr(ElementByElementBlockInverse(vfes, *Mt)); 

	mfem::ParBilinearForm Maform(&fes); 
	mfem::ConstantCoefficient alpha_c(is_half_range ? alpha/2 : alpha); 
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
	if (is_half_range) {
		Dform.AddBdrFaceIntegrator(new DGJumpAverageIntegrator(0.5), marshak_bdr_attrs); 		
	}
	Dform.Assemble(); 
	Dform.Finalize(); 
	D = HypreParMatrixPtr(Dform.ParallelAssemble()); 
	DT = HypreParMatrixPtr(D->Transpose()); 

	auto DiMt = HypreParMatrixPtr(mfem::ParMult(D.get(), iMt.get(), true)); 
	auto DiMtDT = HypreParMatrixPtr(mfem::ParMult(DiMt.get(), DT.get(), true)); 
	S = HypreParMatrixPtr(mfem::ParAdd(DiMtDT.get(), Ma.get())); 
}

BlockIPDiffusionDiscretization::BlockIPDiffusionDiscretization(mfem::ParFiniteElementSpace &_fes, mfem::ParFiniteElementSpace &_vfes, 
	mfem::Coefficient &_total, mfem::Coefficient &_absorption, double _alpha, double _kappa, 
	bool _mip, bool _scale_ip_stabilization, int _reflect_bdr_attr, DiffusionBoundaryConditionType _Jbcs)
	: fes(_fes), vfes(_vfes), total(_total), absorption(_absorption), alpha(_alpha), kappa(_kappa), 
	  mip(_mip), scale_ip_stabilization(_scale_ip_stabilization), reflect_bdr_attr(_reflect_bdr_attr), Jbcs(_Jbcs)
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
	marshak_bdr_attrs.SetSize(mesh_bdr_attributes.Max()); 
	reflect_bdr_attrs.SetSize(mesh_bdr_attributes.Max()); 
	marshak_bdr_attrs = 1; 
	reflect_bdr_attrs = 0; 
	if (reflect_bdr_attr > 0) {
		marshak_bdr_attrs[reflect_bdr_attr-1] = 0; 
		reflect_bdr_attrs[reflect_bdr_attr-1] = 1; 
	}
	const auto dim = mesh.Dimension(); 
	const auto is_half_range = Jbcs == DiffusionBoundaryConditionType::HALF_RANGE; 
	const auto is_half_range_ref = Jbcs == DiffusionBoundaryConditionType::HALF_RANGE_REFLECT; 

	mfem::ParBilinearForm Mtform(&vfes);
	mfem::ProductCoefficient total3(3.0, total); 
	Mtform.AddDomainIntegrator(new mfem::VectorMassIntegrator(total3)); 
	if (is_half_range or is_half_range_ref) {
		Mtform.AddBdrFaceIntegrator(new DGVectorJumpJumpIntegrator(1.0/alpha), reflect_bdr_attrs); 		
	}
	if (is_half_range) {
		Mtform.AddBdrFaceIntegrator(new DGVectorJumpJumpIntegrator(1./2/alpha), marshak_bdr_attrs); 
	}
	Mtform.Assemble(); 
	Mtform.Finalize();  
	auto Mt = HypreParMatrixPtr(Mtform.ParallelAssemble()); 
	iMt = HypreParMatrixPtr(ElementByElementBlockInverse(vfes, *Mt)); 

	mfem::ParBilinearForm Maform(&fes); 
	mfem::ConstantCoefficient alpha_c(is_half_range ? alpha/2 : alpha); 
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
	if (is_half_range) Dform.AddBdrFaceIntegrator(new DGJumpAverageIntegrator(0.5), marshak_bdr_attrs); 
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
	PhaseSpaceCoefficient &inflow_coef, double _alpha, int reflect_bdr_attr, DiffusionBoundaryConditionType _Jbcs)
	: fes(_fes), vfes(_vfes), quad(_quad), psi_ext(_psi_ext), alpha(_alpha), reflect_bdr_attr(reflect_bdr_attr), Jbcs(_Jbcs)
{
	offsets.SetSize(3); 
	offsets[0] = 0; 
	offsets[1] = vfes.GetVSize(); 
	offsets[2] = fes.GetVSize(); 
	offsets.PartialSum(); 

	width = TotalExtent(psi_ext); 
	height = offsets.Last(); 

	const auto &mesh = *fes.GetParMesh(); 
	const auto &bdr_attrs = mesh.bdr_attributes; 
	marshak_bdr_attrs.SetSize(bdr_attrs.Max()); 
	reflect_bdr_attrs.SetSize(bdr_attrs.Max()); 
	marshak_bdr_attrs = 1; 
	reflect_bdr_attrs = 0; 
	if (reflect_bdr_attr > 0) {
		marshak_bdr_attrs[reflect_bdr_attr-1] = 0; 
		reflect_bdr_attrs[reflect_bdr_attr-1] = 1; 
	} 
	const auto is_half_range = Jbcs == DiffusionBoundaryConditionType::HALF_RANGE; 

	const auto dim = fes.GetMesh()->Dimension(); 
	Q0.SetSize(fes.GetVSize()); 
	Q0 = 0.0; 
	Q1.SetSize(vfes.GetVSize()); 
	Q1 = 0.0; 

	for (auto a=0; a<quad.Size(); a++) {
		const auto &Omega = quad.GetOmega(a); 
		source_coef.SetAngle(Omega); 
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
	mfem::ProductCoefficient Jin2(2.0, Jin); 
	mfem::Coefficient *coef; 
	if (is_half_range) coef = &Jin; 
	else coef = &Jin2; 
	mfem::LinearForm fform(&fes); 
	fform.AddBdrFaceIntegrator(new ProjectedCoefBoundaryLFIntegrator(*coef, *fes.FEColl(), 2, 1)); 
	fform.Assemble(); 
	Q0.Add(-1.0, fform); 

	if (is_half_range) {
		mfem::LinearForm gform(&vfes); 
		gform.AddBdrFaceIntegrator(new ProjectedCoefBoundaryNormalLFIntegrator(Jin, *fes.FEColl(), 2, 1)); 		
		gform.Assemble(); 

		Q1.Add(1./alpha/3, gform); 
	}
	Q1 *= 3.0; 		
}

void BlockDiffusionSMMSourceOperator::Mult(const mfem::Vector &psi, mfem::Vector &source) const 
{
	mfem::BlockVector bv(source.GetData(), offsets); 
	ConstTransportVectorView psi_view(psi.GetData(), psi_ext); 
	SMMCorrectionTensorCoefficient T(fes, quad, psi_view); 
	SMMBdrCorrectionFactorCoefficient beta(fes, quad, psi_view, alpha); 	
	const auto is_half_range = Jbcs == DiffusionBoundaryConditionType::HALF_RANGE; 

	mfem::ParLinearForm fform(&fes, bv.GetBlock(1).GetData()); 
	mfem::ProductCoefficient bdr_coef_f(is_half_range ? -0.5 : -1.0, beta); 
	fform.AddBdrFaceIntegrator(new ProjectedCoefBoundaryLFIntegrator(bdr_coef_f, *fes.FEColl(), 2, 1), marshak_bdr_attrs); 
	fform.Assemble(); 

	mfem::ParLinearForm gform(&vfes, bv.GetBlock(0).GetData()); 
	gform.AddDomainIntegrator(new WeakTensorDivergenceLFIntegrator(T)); 
	gform.AddInteriorFaceIntegrator(new VectorJumpTensorAverageLFIntegrator(T)); 
	gform.AddBdrFaceIntegrator(new VectorJumpTensorAverageLFIntegrator(T)); 
	mfem::ProductCoefficient bdr_coef_g(1./2/alpha/3, beta);
	if (is_half_range) {
		gform.AddBdrFaceIntegrator(new ProjectedCoefBoundaryNormalLFIntegrator(bdr_coef_g, *fes.FEColl(), 2, 1), marshak_bdr_attrs); 		
	}
	gform.Assemble();
	gform *= 3.0;  

	fform += Q0; 
	gform += Q1; 
}