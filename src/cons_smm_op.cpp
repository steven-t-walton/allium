#include "cons_smm_op.hpp"
#include "smm_integrators.hpp"
#include "transport_op.hpp"
#include "p1diffusion.hpp"
#include "mip.hpp"
#include "lumping.hpp"

MomentFaceClosuresOperator::MomentFaceClosuresOperator(
	mfem::ParFiniteElementSpace &fes,
	mfem::ParFiniteElementSpace &trace_fes, mfem::ParFiniteElementSpace &trace_vfes, 
	const AngularQuadrature &quad, const TransportVectorExtents &psi_ext, int reflect_bdr_attr)
	: fes(fes), trace_fes(trace_fes), trace_vfes(trace_vfes), 
	  quad(quad), psi_ext(psi_ext), reflect_bdr_attr(reflect_bdr_attr)
{
	width = TotalExtent(psi_ext); 
	height = trace_fes.GetVSize() + trace_vfes.GetVSize(); 
	auto &mesh = *fes.GetParMesh(); 
	auto face_to_element = std::unique_ptr<const mfem::Table>(mesh.GetFaceToAllElementTable()); 
	element_to_face = std::unique_ptr<const mfem::Table>(mfem::Transpose(*face_to_element)); 
	for (auto i=0; i<mesh.GetNBE(); i++) {
		face_to_bdr_el[mesh.GetBdrElementFaceIndex(i)] = i; 
	}
}

void MomentFaceClosuresOperator::Mult(const mfem::Vector &psi, mfem::Vector &closures) const
{
	ConstTransportVectorView psi_view(psi.GetData(), psi_ext);
	mfem::Vector beta(closures, 0, trace_fes.GetVSize()); 
	mfem::Vector tensor(closures, trace_fes.GetVSize(), trace_vfes.GetVSize());
	auto &mesh = *fes.GetParMesh(); 
	const auto dim = mesh.Dimension(); 
	mfem::Vector nor(dim), shape, psi_local; 
	mfem::Array<int> psi_dof, beta_dof, tensor_dof; 
	for (auto e=0; e<fes.GetNE(); e++) {
		const auto &psi_el = *fes.GetFE(e); 
		shape.SetSize(psi_el.GetDof()); 
		psi_local.SetSize(psi_el.GetDof()); 
		fes.GetElementDofs(e, psi_dof);
		trace_fes.GetElementDofs(e, beta_dof);  
		trace_vfes.GetElementVDofs(e, tensor_dof); 
		const auto *faces = element_to_face->GetRow(e); 
		for (auto f=0; f<element_to_face->RowSize(e); f++) {
			bool reflect = false, bdr = false; 
			const auto face = faces[f]; 
			const auto &info = mesh.GetFaceInformation(face); 
			mfem::FaceElementTransformations *trans; 
			if (info.IsShared()) {
				trans = mesh.GetSharedFaceTransformationsByLocalIndex(face, true); 
			} if (info.IsBoundary()) {
				const auto be = face_to_bdr_el.at(face); 
				trans = mesh.GetBdrFaceTransformations(be); 
				reflect = trans->Attribute == reflect_bdr_attr; 
				bdr = true; 
			} else {
				trans = mesh.GetFaceElementTransformations(face); 
			}			
			bool keep_order = e == trans->Elem1No; 
			const auto local_face_id = info.element[keep_order ? 0 : 1].local_face_id; 
			const auto &tr_el = *trace_fes.GetTraceElement(face, mesh.GetFaceGeometry(face)); 
			const auto &ir = tr_el.GetNodes(); 

			const auto ip = mfem::Geometries.GetCenter(trans->GetGeometryType()); 
			trans->SetAllIntPoints(&ip); 
			if (dim==1) {
				nor(0) = 2*trans->GetElement1IntPoint().x - 1.0;
			} else {
				mfem::CalcOrtho(trans->Jacobian(), nor); 				
			}
			nor.Set((keep_order ? 1.0 : -1.0)/nor.Norml2(), nor); 

			mfem::IntegrationPoint eip; 
			for (auto n=0; n<ir.GetNPoints(); n++) {
				const auto &ip = ir.IntPoint(n); 
				trans->SetAllIntPoints(&ip); 
				if (keep_order) {
					eip = trans->GetElement1IntPoint(); 
				} else {
					eip = trans->GetElement2IntPoint(); 
				}

				psi_el.CalcShape(eip, shape); 

				double Jout = 0.0, dot2 = 0.0; 
				mfem::Vector Pout(dim); 
				Pout = 0.0; 
				for (auto a=0; a<quad.Size(); a++) {
					const auto &Omega = quad.GetOmega(a); 
					double w = quad.GetWeight(a); 
					double dot = Omega * nor; 
					if (dot > 0) {
						for (auto i=0; i<psi_dof.Size(); i++) { psi_local[i] = psi_view(0,a,psi_dof[i]); }
						double psi_at_ip = shape * psi_local; 
						dot2 += dot*dot * psi_at_ip * w; // \int_{Omega.n>0} (Omega.n)^2 psi dOmega 
						for (auto d=0; d<dim; d++) {
							Pout(d) += Omega(d) * dot * psi_at_ip * w; // int_{Omega.n>0} Omega \otimes Omega n psi dOmega 
						}
						Jout += dot * psi_at_ip * w; // \int_{Omega.n>0} Omega.n psi dOmega 
					}
				}
				beta(beta_dof[local_face_id*tr_el.GetDof() + n]) = Jout; 
				for (auto d=0; d<dim; d++) {
					double val; 
					if (reflect) {
						val = 2.0*dot2*nor(d); 
					} else {
						val = Pout(d); 
					}							
					tensor(tensor_dof[local_face_id*tr_el.GetDof()*dim + n + d*tr_el.GetDof()]) 
						= val; 
				}
			}
		}
	}
}

void FormTransportSourceMoments(mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &vfes, 
	const AngularQuadrature &quad, ConstTransportVectorView source_vec, mfem::Vector &Q0, mfem::Vector &Q1)
{
	const auto dim = fes.GetMesh()->Dimension(); 
	Q0.SetSize(fes.GetVSize()); 
	Q0 = 0.0; 
	Q1.SetSize(vfes.GetVSize()); 
	Q1 = 0.0; 

	mfem::Array<int> dofs, vdofs; 
	for (auto a=0; a<quad.Size(); a++) {
		const auto &Omega = quad.GetOmega(a); 
		for (auto e=0; e<fes.GetNE(); e++) {
			fes.GetElementDofs(e, dofs); 
			vfes.GetElementVDofs(e, vdofs); 
			for (const auto &dof : dofs) {
				Q0(dof) += source_vec(0,a,dof) * quad.GetWeight(a); 
			}
			for (auto d=0; d<dim; d++) {
				for (auto i=0; i<dofs.Size(); i++) {
					Q1(vdofs[i + dofs.Size()*d]) += Omega(d) * source_vec(0,a,dofs[i]) * quad.GetWeight(a); 
				}
			}
		}
	}
}

ConsistentSMMSourceOperatorBase::ConsistentSMMSourceOperatorBase(
	mfem::ParFiniteElementSpace &_fes, mfem::ParFiniteElementSpace &_vfes, 
	const AngularQuadrature &_quad, const TransportVectorExtents &_psi_ext, 
	ConstTransportVectorView source_vec, double _alpha, int _reflect_bdr_attr, int _lumping)
	: fes(_fes), vfes(_vfes), quad(_quad), psi_ext(_psi_ext), 
	  alpha(_alpha), reflect_bdr_attr(_reflect_bdr_attr), lumping(_lumping)
{
	offsets.SetSize(3); 
	offsets[0] = 0; 
	offsets[1] = vfes.GetVSize(); 
	offsets[2] = fes.GetVSize(); 
	offsets.PartialSum(); 

	width = TotalExtent(psi_ext); 
	height = offsets.Last(); 
	const auto dim = fes.GetMesh()->Dimension(); 

	FormTransportSourceMoments(fes, vfes, quad, source_vec, Q0, Q1); 
	Q1 *= 3.0; 

	auto &mesh = *fes.GetParMesh(); 

	if (fes.IsVariableOrder()) { MFEM_ABORT("variable order not supported for consistent SMM"); }
	trace_coll = std::make_unique<DGTrace_FECollection>(fes.GetOrder(0), dim); 
	trace_fes = std::make_unique<mfem::ParFiniteElementSpace>(&mesh, trace_coll.get()); 
	trace_vfes = std::make_unique<mfem::ParFiniteElementSpace>(&mesh, trace_coll.get(), dim); 
	face_closure_op = std::make_unique<MomentFaceClosuresOperator>(
		fes, *trace_fes, *trace_vfes, quad, psi_ext, reflect_bdr_attr);
	face_closure_data.SetSize(face_closure_op->Height());

	const auto &mesh_bdr_attributes = mesh.bdr_attributes; 
	marshak_bdr_attrs.SetSize(mesh_bdr_attributes.Max());
	reflect_bdr_attrs.SetSize(mesh_bdr_attributes.Max()); 
	marshak_bdr_attrs = 1; 
	reflect_bdr_attrs = 0; 
	if (reflect_bdr_attr > 0) {
		marshak_bdr_attrs[reflect_bdr_attr-1] = 0; 
		reflect_bdr_attrs[reflect_bdr_attr-1] = 1; 
	}
}

void ConsistentSMMSourceOperatorBase::Mult(const mfem::Vector &psi, mfem::Vector &source) const 
{
	const bool lump_grad = IsGradientLumped(lumping);
	const bool lump_face = IsFaceLumped(lumping);

	mfem::BlockVector bv(source.GetData(), offsets); 
	ConstTransportVectorView psi_view(psi.GetData(), psi_ext); 
	SecondMomentTensorCoefficient P(fes, quad, psi_view); 
	face_closure_op->Mult(psi, face_closure_data);
	mfem::ParGridFunction beta(trace_fes.get(), face_closure_data, 0); 
	mfem::ParGridFunction tensor(trace_vfes.get(), face_closure_data, trace_fes->GetVSize());
	beta.ExchangeFaceNbrData(); 
	tensor.ExchangeFaceNbrData();

	mfem::ParLinearForm fform(&fes, bv.GetBlock(1).GetData());
	if (lump_face) {
		fform.AddInteriorFaceIntegrator(new QuadratureLumpedLFIntegrator(new CSMMZerothMomentFaceLFIntegrator(beta)));
		fform.AddBdrFaceIntegrator(new QuadratureLumpedLFIntegrator(new CSMMZerothMomentFaceLFIntegrator(beta)), 
			marshak_bdr_attrs); 		
	} else {
		fform.AddInteriorFaceIntegrator(new CSMMZerothMomentFaceLFIntegrator(beta));
		fform.AddBdrFaceIntegrator(new CSMMZerothMomentFaceLFIntegrator(beta), marshak_bdr_attrs); 		
	}
	fform.Assemble(); 

	mfem::ParLinearForm gform(&vfes, bv.GetBlock(0).GetData()); 
	if (lump_grad) {
		gform.AddDomainIntegrator(new QuadratureLumpedLFIntegrator(new WeakTensorDivergenceLFIntegrator(P))); 
	} else {
		gform.AddDomainIntegrator(new WeakTensorDivergenceLFIntegrator(P)); 
	}
	if (lump_face) {
		gform.AddInteriorFaceIntegrator(new QuadratureLumpedLFIntegrator(new CSMMFirstMomentFaceLFIntegrator(tensor))); 
		gform.AddBdrFaceIntegrator(new QuadratureLumpedLFIntegrator(new CSMMFirstMomentFaceLFIntegrator(tensor))); 				
	} else {
		gform.AddInteriorFaceIntegrator(new CSMMFirstMomentFaceLFIntegrator(tensor)); 
		gform.AddBdrFaceIntegrator(new CSMMFirstMomentFaceLFIntegrator(tensor)); 		
	}
	gform.Assemble(); 
	gform *= 3.0; 

	fform += Q0; 
	gform += Q1; 
}

ConsistentSMMSourceOperator::ConsistentSMMSourceOperator(mfem::ParFiniteElementSpace &_fes, mfem::ParFiniteElementSpace &_vfes, 
		const AngularQuadrature &_quad, const TransportVectorExtents &_psi_ext, 
		ConstTransportVectorView source_vec, double _alpha, int _reflect_bdr_attr)
	: ConsistentSMMSourceOperatorBase(_fes, _vfes, _quad, _psi_ext, source_vec, _alpha, _reflect_bdr_attr)
{
	const auto dim = quad.Dimension(); 
	phi_ext = MomentVectorExtents(1,dim+1,fes.GetVSize()); 
	moments.SetSize(TotalExtent(phi_ext)); 

	mfem::ParBilinearForm F11form(&vfes); 
	F11form.AddInteriorFaceIntegrator(new DGVectorJumpJumpIntegrator(1.0/2/alpha)); 
	F11form.AddBdrFaceIntegrator(new DGVectorJumpJumpIntegrator(1.0/2/alpha), marshak_bdr_attrs);
	F11form.AddBdrFaceIntegrator(new DGVectorJumpJumpIntegrator(1.0/alpha), reflect_bdr_attrs);  
	F11form.Assemble(); 
	F11form.Finalize(); 
	F11 = HypreParMatrixPtr(F11form.ParallelAssemble()); 

	mfem::ParMixedBilinearForm Dform(&vfes, &fes); 
	mfem::ConstantCoefficient neg_one(-1.0); 
	Dform.AddDomainIntegrator(new mfem::TransposeIntegrator(new mfem::GradientIntegrator(neg_one))); 
	Dform.AddInteriorFaceIntegrator(new DGJumpAverageIntegrator); 
	Dform.AddBdrFaceIntegrator(new DGJumpAverageIntegrator(0.5), marshak_bdr_attrs); 
	Dform.Assemble(); 
	Dform.Finalize(); 
	D = HypreParMatrixPtr(Dform.ParallelAssemble()); 

	mfem::ParMixedBilinearForm F21form(&vfes, &fes); 
	F21form.AddInteriorFaceIntegrator(new DGJumpAverageIntegrator); 
	F21form.AddBdrFaceIntegrator(new DGJumpAverageIntegrator(0.5), marshak_bdr_attrs); 
	F21form.Assemble(); 
	F21form.Finalize(); 
	F21 = HypreParMatrixPtr(F21form.ParallelAssemble()); 	

	mfem::ParBilinearForm F22form(&fes); 
	mfem::ConstantCoefficient alpha_c(alpha/2); 
	F22form.AddInteriorFaceIntegrator(new PenaltyIntegrator(alpha/2, false)); 
	F22form.AddBdrFaceIntegrator(new mfem::BoundaryMassIntegrator(alpha_c), marshak_bdr_attrs); 
	F22form.Assemble(); 
	F22form.Finalize(); 
	F22 = HypreParMatrixPtr(F22form.ParallelAssemble()); 
}

void ConsistentSMMSourceOperator::Mult(const mfem::Vector &psi, mfem::Vector &source) const 
{
	ConsistentSMMSourceOperatorBase::Mult(psi, source); 
	mfem::BlockVector bv(source.GetData(), offsets); 

	const auto dim = quad.Dimension(); 
	DiscreteToMoment D(quad, psi_ext, phi_ext); 
	D.Mult(psi, moments); 
	mfem::Vector phi(moments, 0, fes.GetVSize()); 
	mfem::Vector J(moments, fes.GetVSize(), dim*fes.GetVSize()); 

	F11->Mult(1.0, J, 1.0, bv.GetBlock(0)); 
	F21->Mult(1.0, J, 1.0, bv.GetBlock(1)); 
	this->D->MultTranspose(-1.0, phi, 1.0, bv.GetBlock(0)); 
	F22->Mult(1.0, phi, 1.0, bv.GetBlock(1)); 
}

ConsistentLDGSMMSourceOperator::ConsistentLDGSMMSourceOperator(const BlockLDGDiffusionDiscretization &_lhs, 
	const AngularQuadrature &_quad, const TransportVectorExtents &_psi_ext, ConstTransportVectorView source_vec)
	: lhs(_lhs), ConsistentSMMSourceOperatorBase(_lhs.fes, _lhs.vfes, _quad, _psi_ext, source_vec, _lhs.alpha, _lhs.reflect_bdr_attr)
{
	const auto dim = quad.Dimension(); 
	phi_ext = MomentVectorExtents(1,dim+1,fes.GetVSize()); 
	moments.SetSize(TotalExtent(phi_ext)); 

	const auto &alpha = lhs.alpha; 
	const bool is_half_range = lhs.Jbcs == DiffusionBoundaryConditionType::HALF_RANGE; 
	const bool is_half_range_ref = lhs.Jbcs == DiffusionBoundaryConditionType::HALF_RANGE_REFLECT; 

	if (is_half_range or is_half_range_ref) {
		mfem::ParBilinearForm Mtform(&vfes);
		if (is_half_range) 
			Mtform.AddBdrFaceIntegrator(new DGVectorJumpJumpIntegrator(1./2/alpha), marshak_bdr_attrs); 
		if (is_half_range or is_half_range_ref)
			Mtform.AddBdrFaceIntegrator(new DGVectorJumpJumpIntegrator(1.0/alpha), reflect_bdr_attrs); 		
		Mtform.Assemble(); 
		Mtform.Finalize();  
		Mt = HypreParMatrixPtr(Mtform.ParallelAssemble()); 
	}

	mfem::ParBilinearForm Maform(&fes); 
	mfem::ConstantCoefficient alpha_c(is_half_range ? alpha/2 : alpha); 
	Maform.AddInteriorFaceIntegrator(new PenaltyIntegrator(alpha/2, false)); 
	Maform.AddBdrFaceIntegrator(new mfem::BoundaryMassIntegrator(alpha_c), marshak_bdr_attrs); 
	Maform.Assemble(); 
	Maform.Finalize(); 
	Ma = HypreParMatrixPtr(Maform.ParallelAssemble());

	mfem::ParMixedBilinearForm Dform(&vfes, &fes); 
	mfem::RatioCoefficient diffco(1.0/3, lhs.total); 
	if (lhs.scale_ldg_stabilization) {
		double kappa = pow(fes.GetOrder(0)+1, 2); 
		Dform.AddInteriorFaceIntegrator(new LDGTraceIntegrator(diffco, lhs.beta, kappa, alpha/2)); 
	} else {
		Dform.AddInteriorFaceIntegrator(new mfem::LDGTraceIntegrator(&lhs.beta)); 		
	}
	if (is_half_range) {
		Dform.AddBdrFaceIntegrator(new DGJumpAverageIntegrator(0.5), lhs.marshak_bdr_attrs); 		
	}
	Dform.Assemble(); 
	Dform.Finalize(); 
	D = HypreParMatrixPtr(Dform.ParallelAssemble()); 
}

void ConsistentLDGSMMSourceOperator::Mult(const mfem::Vector &psi, mfem::Vector &source) const 
{
	ConsistentSMMSourceOperatorBase::Mult(psi, source); 
	mfem::BlockVector bv(source.GetData(), offsets); 

	const auto dim = quad.Dimension(); 
	DiscreteToMoment D(quad, psi_ext, phi_ext); 
	D.Mult(psi, moments); 
	mfem::Vector phi(moments, 0, fes.GetVSize()); 
	mfem::Vector J(moments, fes.GetVSize(), dim*fes.GetVSize()); 
	if (Mt) Mt->Mult(1.0, J, 1.0, bv.GetBlock(0)); 
	this->D->Mult(1.0, J, 1.0, bv.GetBlock(1)); 
	lhs.DT->Mult(-1.0, phi, 1.0, bv.GetBlock(0)); 
	Ma->Mult(1.0, phi, 1.0, bv.GetBlock(1)); 
}

ConsistentIPSMMSourceOperator::ConsistentIPSMMSourceOperator(const BlockIPDiffusionDiscretization &_lhs, 
	const AngularQuadrature &_quad, const TransportVectorExtents &_psi_ext, ConstTransportVectorView source_vec)
	: lhs(_lhs), ConsistentSMMSourceOperatorBase(_lhs.fes, _lhs.vfes, _quad, _psi_ext, source_vec, _lhs.alpha, _lhs.reflect_bdr_attr)
{
	const auto dim = quad.Dimension(); 
	phi_ext = MomentVectorExtents(1,dim+1,fes.GetVSize()); 
	moments.SetSize(TotalExtent(phi_ext)); 

	const auto &alpha = lhs.alpha;
	const auto &kappa = lhs.kappa; 
	const auto is_half_range = lhs.Jbcs == DiffusionBoundaryConditionType::HALF_RANGE; 
	const auto is_half_range_ref = lhs.Jbcs == DiffusionBoundaryConditionType::HALF_RANGE_REFLECT;

	if (is_half_range or is_half_range_ref) {
		mfem::ParBilinearForm Mtform(&vfes);
		if (is_half_range) 
			Mtform.AddBdrFaceIntegrator(new DGVectorJumpJumpIntegrator(1./2/alpha), marshak_bdr_attrs); 
		if (is_half_range or is_half_range_ref) 
			Mtform.AddBdrFaceIntegrator(new DGVectorJumpJumpIntegrator(1.0/alpha), reflect_bdr_attrs); 
		Mtform.Assemble(); 
		Mtform.Finalize();  
		Mt = HypreParMatrixPtr(Mtform.ParallelAssemble()); 		
	}

	mfem::ParBilinearForm Maform(&fes); 
	mfem::ConstantCoefficient alpha_c(is_half_range ? alpha/2 : alpha); 
	mfem::RatioCoefficient diffco(1./3, lhs.total); 
	mfem::Coefficient *coef = lhs.scale_ip_stabilization ? &diffco : nullptr; 
	double limit = lhs.mip ? alpha/2 : 0.0; 
	Maform.AddInteriorFaceIntegrator(new PenaltyIntegrator(kappa, limit, coef)); 		
	Maform.AddBdrFaceIntegrator(new mfem::BoundaryMassIntegrator(alpha_c), marshak_bdr_attrs); 
	Maform.Assemble(); 
	Maform.Finalize(); 
	Ma = HypreParMatrixPtr(Maform.ParallelAssemble());

	mfem::ParMixedBilinearForm Dform(&vfes, &fes); 
	Dform.AddInteriorFaceIntegrator(new DGJumpAverageIntegrator); 
	if (is_half_range) Dform.AddBdrFaceIntegrator(new DGJumpAverageIntegrator(0.5), marshak_bdr_attrs); 
	Dform.Assemble(); 
	Dform.Finalize(); 
	D = HypreParMatrixPtr(Dform.ParallelAssemble()); 
}

void ConsistentIPSMMSourceOperator::Mult(const mfem::Vector &psi, mfem::Vector &source) const 
{
	ConsistentSMMSourceOperatorBase::Mult(psi, source); 
	mfem::BlockVector bv(source.GetData(), offsets); 

	const auto dim = quad.Dimension(); 
	DiscreteToMoment D(quad, psi_ext, phi_ext); 
	D.Mult(psi, moments); 
	mfem::Vector phi(moments, 0, fes.GetVSize()); 
	mfem::Vector J(moments, fes.GetVSize(), dim*fes.GetVSize()); 
	if (Mt) Mt->Mult(1.0, J, 1.0, bv.GetBlock(0)); 
	this->D->Mult(1.0, J, 1.0, bv.GetBlock(1)); 
	lhs.DT->Mult(-1.0, phi, 1.0, bv.GetBlock(0)); 
	Ma->Mult(1.0, phi, 1.0, bv.GetBlock(1));
}
