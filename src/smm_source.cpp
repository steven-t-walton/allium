#include "smm_source.hpp"
#include "transport_op.hpp"
#include "lumping.hpp"
#include "moment_integrators.hpp"
#include "dg_trace_coll.hpp"
#include "smm_integrators.hpp"
#include "smm_coef.hpp"
#include "coefficient.hpp"

MomentFaceClosuresOperator::MomentFaceClosuresOperator(
	mfem::ParFiniteElementSpace &fes,
	mfem::ParFiniteElementSpace &trace_fes, mfem::ParFiniteElementSpace &trace_vfes, 
	const AngularQuadrature &quad, const TransportVectorExtents &psi_ext, const BoundaryConditionMap &bc_map)
	: fes(fes), trace_fes(trace_fes), trace_vfes(trace_vfes), 
	  quad(quad), psi_ext(psi_ext), bc_map(bc_map)
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
				reflect = bc_map.at(trans->Attribute) == BoundaryCondition::REFLECTIVE;
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

ConsistentSMMOperatorBase::ConsistentSMMOperatorBase(mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &vfes, 
	const AngularQuadrature &quad, const TransportVectorExtents &psi_ext, const BoundaryConditionMap &bc_map, 
	const mfem::Vector &source_vec, double alpha, int lumping)
	: fes(fes), vfes(vfes), quad(quad), psi_ext(psi_ext), bc_map(bc_map), alpha(alpha), lumping(lumping)
{
	width = TotalExtent(psi_ext);
	height = fes.GetVSize() + vfes.GetVSize();

	auto &mesh = *fes.GetParMesh(); 
	const auto dim = mesh.Dimension();

	phi_ext = MomentVectorExtents(1, dim+1, fes.GetVSize());
	Q.SetSize(TotalExtent(phi_ext));
	Q0.MakeRef(Q, 0, fes.GetVSize());
	Q1.MakeRef(Q, fes.GetVSize(), vfes.GetVSize());
	DiscreteToMoment d2m(quad, psi_ext, phi_ext);
	d2m.Mult(source_vec, Q);
	Q1 *= 3.0;

	if (fes.IsVariableOrder()) { MFEM_ABORT("variable order not supported for consistent SMM"); }
	trace_coll = std::make_unique<DGTrace_FECollection>(fes.GetOrder(0), dim); 
	trace_fes = std::make_unique<mfem::ParFiniteElementSpace>(&mesh, trace_coll.get()); 
	trace_vfes = std::make_unique<mfem::ParFiniteElementSpace>(&mesh, trace_coll.get(), dim); 
	face_closure_op = std::make_unique<MomentFaceClosuresOperator>(
		fes, *trace_fes, *trace_vfes, quad, psi_ext, bc_map);
	face_closure_data.SetSize(face_closure_op->Height());
	beta.MakeRef(trace_fes.get(), face_closure_data, 0);
	tensor.MakeRef(trace_vfes.get(), face_closure_data, trace_fes->GetVSize());

	marshak_bdr_attrs = CreateBdrAttributeMarker<INFLOW>(bc_map);
	reflect_bdr_attrs = CreateBdrAttributeMarker<REFLECTIVE>(bc_map);
}

void ConsistentSMMOperatorBase::Mult(const mfem::Vector &psi, mfem::Vector &source) const
{
	const bool lump_grad = IsGradientLumped(lumping);
	const bool lump_face = IsFaceLumped(lumping);

	ConstTransportVectorView psi_view(psi.GetData(), psi_ext); 
	SecondMomentTensorCoefficient P(fes, quad, psi_view); 
	face_closure_op->Mult(psi, face_closure_data);
	beta.ExchangeFaceNbrData(); 
	tensor.ExchangeFaceNbrData();

	mfem::ParLinearForm fform(&fes, source.GetData() + vfes.GetVSize());
	if (lump_face) {
		fform.AddInteriorFaceIntegrator(new QuadratureLumpedLFIntegrator(new CSMMZerothMomentFaceLFIntegrator(beta)));
		fform.AddBdrFaceIntegrator(new QuadratureLumpedLFIntegrator(new CSMMZerothMomentFaceLFIntegrator(beta)), 
			marshak_bdr_attrs); 		
	} else {
		fform.AddInteriorFaceIntegrator(new CSMMZerothMomentFaceLFIntegrator(beta));
		fform.AddBdrFaceIntegrator(new CSMMZerothMomentFaceLFIntegrator(beta), marshak_bdr_attrs); 		
	}
	fform.Assemble(); 

	mfem::ParLinearForm gform(&vfes, source.GetData()); 
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

ConsistentP1SMMOperator::ConsistentP1SMMOperator(const P1Discretization &disc,
	const AngularQuadrature &quad, const TransportVectorExtents &psi_ext, const mfem::Vector &source_vec)
	: ConsistentSMMOperatorBase(disc.fes, disc.vfes, quad, psi_ext, disc.bc_map, source_vec, disc.alpha, disc.lumping)
{
	moments.SetSize(TotalExtent(phi_ext));

	const bool lump_mass = IsMassLumped(lumping);
	const bool lump_grad = IsGradientLumped(lumping);
	const bool lump_face = IsFaceLumped(lumping);

	mfem::BilinearFormIntegrator *bfi;

	mfem::ParBilinearForm F11form(&vfes); 
	bfi = new DGVectorJumpJumpIntegrator(1.0/2/alpha);
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi);
	F11form.AddInteriorFaceIntegrator(bfi); 
	bfi = new DGVectorJumpJumpIntegrator(1.0/2/alpha);
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi);
	F11form.AddBdrFaceIntegrator(bfi, marshak_bdr_attrs);
	bfi = new DGVectorJumpJumpIntegrator(1.0/alpha);
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi);
	F11form.AddBdrFaceIntegrator(bfi, reflect_bdr_attrs);  
	F11form.Assemble(); 
	F11form.Finalize(); 
	F11 = HypreParMatrixPtr(F11form.ParallelAssemble()); 

	mfem::ParMixedBilinearForm Dform(&vfes, &fes); 
	mfem::ConstantCoefficient neg_one(-1.0); 
	bfi = new mfem::TransposeIntegrator(new mfem::GradientIntegrator(neg_one));
	if (lump_grad) bfi = new QuadratureLumpedIntegrator(bfi);
	Dform.AddDomainIntegrator(bfi);
	bfi = new DGJumpAverageIntegrator;
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi); 
	Dform.AddInteriorFaceIntegrator(bfi); 
	bfi = new DGJumpAverageIntegrator(0.5);
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi);
	Dform.AddBdrFaceIntegrator(bfi, marshak_bdr_attrs); 
	Dform.Assemble(); 
	Dform.Finalize(); 
	D = HypreParMatrixPtr(Dform.ParallelAssemble()); 

	mfem::ParMixedBilinearForm F21form(&vfes, &fes); 
	bfi = new DGJumpAverageIntegrator;
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi);
	F21form.AddInteriorFaceIntegrator(bfi); 
	bfi = new DGJumpAverageIntegrator(0.5);
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi);
	F21form.AddBdrFaceIntegrator(bfi, marshak_bdr_attrs); 
	F21form.Assemble(); 
	F21form.Finalize(); 
	F21 = HypreParMatrixPtr(F21form.ParallelAssemble()); 	

	mfem::ParBilinearForm F22form(&fes); 
	mfem::ConstantCoefficient alpha_c(alpha/2); 
	bfi = new PenaltyIntegrator(alpha/2, false);
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi);
	F22form.AddInteriorFaceIntegrator(bfi); 
	bfi = new mfem::BoundaryMassIntegrator(alpha_c);
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi);
	F22form.AddBdrFaceIntegrator(bfi, marshak_bdr_attrs); 
	F22form.Assemble(); 
	F22form.Finalize(); 
	F22 = HypreParMatrixPtr(F22form.ParallelAssemble()); 
}

void ConsistentP1SMMOperator::Mult(const mfem::Vector &psi, mfem::Vector &source) const
{
	ConsistentSMMOperatorBase::Mult(psi, source); 
	DiscreteToMoment d2m(quad, psi_ext, phi_ext); 
	d2m.Mult(psi, moments); 
	mfem::Vector phi(moments, 0, fes.GetVSize()); 
	mfem::Vector J(moments, fes.GetVSize(), vfes.GetVSize()); 

	mfem::Vector source_J(source, 0, vfes.GetVSize());
	mfem::Vector source_phi(source, vfes.GetVSize(), fes.GetVSize());

	F11->Mult(1.0, J, 1.0, source_J); 
	F21->Mult(1.0, J, 1.0, source_phi); 
	D->MultTranspose(-1.0, phi, 1.0, source_J); 
	F22->Mult(1.0, phi, 1.0, source_phi); 
}

ConsistentLDGSMMOperator::ConsistentLDGSMMOperator(const BlockLDGDiscretization &disc, 
	const AngularQuadrature &quad, const TransportVectorExtents &psi_ext, const mfem::Vector &source_vec)
	: ConsistentSMMOperatorBase(disc.fes, disc.vfes, quad, psi_ext, disc.bc_map, source_vec, disc.alpha, disc.lumping), 
	  beta(disc.beta)
{
	moments.SetSize(TotalExtent(phi_ext));

	const bool lump_mass = IsMassLumped(lumping);
	const bool lump_grad = IsGradientLumped(lumping);
	const bool lump_face = IsFaceLumped(lumping);

	mfem::BilinearFormIntegrator *bfi;

	mfem::ParBilinearForm Maform(&fes); 
	mfem::ConstantCoefficient alpha_c(alpha); 
	bfi = new PenaltyIntegrator(alpha/2, false);
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi);
	Maform.AddInteriorFaceIntegrator(bfi);
	bfi = new mfem::BoundaryMassIntegrator(alpha_c); 
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi);
	Maform.AddBdrFaceIntegrator(bfi, marshak_bdr_attrs); 
	Maform.Assemble(); 
	Maform.Finalize(); 
	Ma = HypreParMatrixPtr(Maform.ParallelAssemble());

	mfem::ParMixedBilinearForm Dform(&vfes, &fes); 
	bfi = new mfem::LDGTraceIntegrator(&beta);
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi);
	Dform.AddInteriorFaceIntegrator(bfi); 		
	Dform.Assemble(); 
	Dform.Finalize(); 
	D = HypreParMatrixPtr(Dform.ParallelAssemble()); 

	mfem::ParMixedBilinearForm Gform(&fes, &vfes); 
	mfem::ConstantCoefficient neg_one(-1.0); 
	bfi = new mfem::GradientIntegrator(neg_one);
	if (lump_grad) bfi = new QuadratureLumpedIntegrator(bfi);
	Gform.AddDomainIntegrator(bfi); 
	bfi = new mfem::TransposeIntegrator(new mfem::LDGTraceIntegrator(&beta)); 
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi);
	Gform.AddInteriorFaceIntegrator(bfi); 		
	Gform.Assemble(); 
	Gform.Finalize(); 
	DT = HypreParMatrixPtr(Gform.ParallelAssemble()); 
}

void ConsistentLDGSMMOperator::Mult(const mfem::Vector &psi, mfem::Vector &source) const
{
	ConsistentSMMOperatorBase::Mult(psi, source); 

	DiscreteToMoment d2m(quad, psi_ext, phi_ext); 
	d2m.Mult(psi, moments); 
	mfem::Vector phi(moments, 0, fes.GetVSize()); 
	mfem::Vector J(moments, fes.GetVSize(), vfes.GetVSize()); 

	mfem::Vector source_J(source, 0, vfes.GetVSize());
	mfem::Vector source_phi(source, vfes.GetVSize(), fes.GetVSize());

	D->Mult(1.0, J, 1.0, source_phi); 
	DT->Mult(-1.0, phi, 1.0, source_J); 
	Ma->Mult(1.0, phi, 1.0, source_phi); 
}

ConsistentIPSMMOperator::ConsistentIPSMMOperator(const BlockIPDiscretization &disc, mfem::Coefficient &total, 
	const AngularQuadrature &quad, const TransportVectorExtents &psi_ext, const mfem::Vector &source_vec)
	: ConsistentSMMOperatorBase(disc.fes, disc.vfes, quad, psi_ext, disc.bc_map, source_vec, disc.alpha, disc.lumping), 
	  total(total), kappa(disc.kappa), mip_val(disc.mip_val), scale_penalty(disc.scale_penalty)
{
	moments.SetSize(TotalExtent(phi_ext));

	const bool lump_mass = IsMassLumped(lumping);
	const bool lump_grad = IsGradientLumped(lumping);
	const bool lump_face = IsFaceLumped(lumping);

	mfem::BilinearFormIntegrator *bfi;

	mfem::ParBilinearForm Maform(&fes); 
	mfem::ConstantCoefficient alpha_c(alpha); 
	mfem::RatioCoefficient diffco(1./3, total); 
	mfem::Coefficient *coef = scale_penalty ? &diffco : nullptr; 
	bfi = new PenaltyIntegrator(kappa, mip_val, coef);
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi);
	Maform.AddInteriorFaceIntegrator(bfi); 		
	bfi = new mfem::BoundaryMassIntegrator(alpha_c);
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi);
	Maform.AddBdrFaceIntegrator(bfi, marshak_bdr_attrs); 
	Maform.Assemble(); 
	Maform.Finalize(); 
	Ma = HypreParMatrixPtr(Maform.ParallelAssemble());

	mfem::ParMixedBilinearForm Dform(&vfes, &fes); 
	bfi = new DGJumpAverageIntegrator;
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi);
	Dform.AddInteriorFaceIntegrator(bfi); 
	Dform.Assemble(); 
	Dform.Finalize(); 
	D = HypreParMatrixPtr(Dform.ParallelAssemble()); 

	mfem::ParMixedBilinearForm Gform(&fes, &vfes); 
	mfem::ConstantCoefficient neg_one(-1.0); 
	bfi = new mfem::GradientIntegrator(neg_one);
	if (lump_grad) bfi = new QuadratureLumpedIntegrator(bfi);
	Gform.AddDomainIntegrator(bfi); 
	bfi = new mfem::TransposeIntegrator(new DGJumpAverageIntegrator); 
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi);
	Gform.AddInteriorFaceIntegrator(bfi); 		
	Gform.Assemble(); 
	Gform.Finalize(); 
	DT = HypreParMatrixPtr(Gform.ParallelAssemble()); 
}

void ConsistentIPSMMOperator::Mult(const mfem::Vector &psi, mfem::Vector &source) const
{
	ConsistentSMMOperatorBase::Mult(psi, source);

	DiscreteToMoment d2m(quad, psi_ext, phi_ext); 
	d2m.Mult(psi, moments); 
	mfem::Vector phi(moments, 0, fes.GetVSize()); 
	mfem::Vector J(moments, fes.GetVSize(), vfes.GetVSize()); 

	mfem::Vector source_J(source, 0, vfes.GetVSize());
	mfem::Vector source_phi(source, vfes.GetVSize(), fes.GetVSize());

	D->Mult(1.0, J, 1.0, source_phi); 
	DT->Mult(-1.0, phi, 1.0, source_J); 
	Ma->Mult(1.0, phi, 1.0, source_phi);
}

IndependentSMMOperator::IndependentSMMOperator(
	mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &tfes, 
	const AngularQuadrature &quad, const TransportVectorExtents &psi_ext, 
	const MultiGroupEnergyGrid &energy, mfem::Coefficient &total, 
	PhaseSpaceCoefficient &source_coef, PhaseSpaceCoefficient &inflow_coef, 
	double alpha, const BoundaryConditionMap &bc_map, int lumping)
	: fes(fes), tfes(tfes), quad(quad), psi_ext(psi_ext), total(total), source_coef(source_coef), 
	  inflow_coef(inflow_coef), alpha(alpha), bc_map(bc_map), lumping(lumping)
{
	width = TotalExtent(psi_ext);
	height = fes.GetTrueVSize();

	marshak_bdr_attrs = CreateBdrAttributeMarker<INFLOW>(bc_map);
	reflect_bdr_attrs = CreateBdrAttributeMarker<REFLECTIVE>(bc_map);

	Q.SetSize(fes.GetVSize());
	Q = 0.0;

	for (int g=0; g<energy.Size(); g++) {
		source_coef.SetEnergy(energy.LowerBound(g), energy.UpperBound(g), energy.MeanEnergy(g));
		for (auto a=0; a<quad.Size(); a++) {
			const auto &Omega = quad.GetOmega(a);
			source_coef.SetAngle(Omega);
			mfem::LinearForm fform(&fes);
			fform.AddDomainIntegrator(new mfem::DomainLFIntegrator(source_coef));
			mfem::VectorConstantCoefficient Omega_coef(Omega); 
			mfem::RatioCoefficient source_total(source_coef, total);
			mfem::ScalarVectorProductCoefficient Omega_source(source_total, Omega_coef);
			fform.AddDomainIntegrator(new mfem::DomainLFGradIntegrator(Omega_source));
			fform.Assemble();
			Q.Add(quad.GetWeight(a), fform);
		}		
	}

	InflowPartialCurrentCoefficient Jin_mg(inflow_coef, quad, energy); 
	VectorComponentSumCoefficient Jin(Jin_mg);
	mfem::ProductCoefficient Jin2(2.0, Jin); 
	mfem::LinearForm fform(&fes); 
	if (IsFaceLumped(lumping))
		fform.AddBdrFaceIntegrator(
			new QuadratureLumpedLFIntegrator(new ProjectedCoefBoundaryLFIntegrator(Jin2, *tfes.FEColl(), 2, 1))); 
	else		
		fform.AddBdrFaceIntegrator(new ProjectedCoefBoundaryLFIntegrator(Jin2, *tfes.FEColl(), 2, 1)); 
	fform.Assemble(); 
	Q.Add(-1.0, fform); 
}

void IndependentSMMOperator::Mult(const mfem::Vector &psi, mfem::Vector &source) const
{
	const bool lump_mass = IsMassLumped(lumping);
	const bool lump_grad = IsGradientLumped(lumping);
	const bool lump_face = IsFaceLumped(lumping);

	ConstTransportVectorView psi_view(psi.GetData(), psi_ext); 
	SMMCorrectionTensorCoefficient T(tfes, quad, psi_view); 
	SMMBdrCorrectionFactorCoefficient beta(tfes, quad, psi_view, alpha);
	MatrixDivergenceGridFunctionCoefficient divT(T);
	mfem::RatioCoefficient neg_total_inv(-1.0, total);
	mfem::ScalarVectorProductCoefficient divT_total(neg_total_inv, divT);

	mfem::ParLinearForm fform(&fes); 
	mfem::ProductCoefficient bdr_coef_f(-1.0, beta); 
	mfem::LinearFormIntegrator *lfi;
	lfi = new ProjectedCoefBoundaryLFIntegrator(bdr_coef_f, *fes.FEColl(), 2, 1);
	if (lump_face) lfi = new QuadratureLumpedLFIntegrator(lfi);
	fform.AddBdrFaceIntegrator(lfi, marshak_bdr_attrs);
	lfi = new mfem::DomainLFGradIntegrator(divT_total);
	if (lump_grad) lfi = new QuadratureLumpedLFIntegrator(lfi);
	fform.AddDomainIntegrator(lfi);
	lfi = new GradAverageTensorJumpLFIntegrator(total, T, 2, 1);
	if (lump_face) lfi = new QuadratureLumpedLFIntegrator(lfi);
	fform.AddInteriorFaceIntegrator(lfi);
	fform.Assemble(); 
	fform += Q;
	fform.ParallelAssemble(source);
}

IndependentBlockSMMOperator::IndependentBlockSMMOperator(
	mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &vfes, 
	const AngularQuadrature &quad, const TransportVectorExtents &psi_ext, 
	PhaseSpaceCoefficient &source_coef, PhaseSpaceCoefficient &inflow_coef, 
	double alpha, const BoundaryConditionMap &bc_map, int lumping)
	: fes(fes), vfes(vfes), quad(quad), psi_ext(psi_ext), source_coef(source_coef), inflow_coef(inflow_coef), 
	  alpha(alpha), bc_map(bc_map), lumping(lumping)
{ 
	width = TotalExtent(psi_ext);
	height = fes.GetVSize() + vfes.GetVSize();

	marshak_bdr_attrs = CreateBdrAttributeMarker<INFLOW>(bc_map);
	reflect_bdr_attrs = CreateBdrAttributeMarker<REFLECTIVE>(bc_map);

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

	GrayInflowPartialCurrentCoefficient Jin(inflow_coef, quad); 
	mfem::ProductCoefficient Jin2(2.0, Jin); 
	mfem::LinearForm fform(&fes); 
	if (IsFaceLumped(lumping))
		fform.AddBdrFaceIntegrator(
			new QuadratureLumpedLFIntegrator(new ProjectedCoefBoundaryLFIntegrator(Jin2, *fes.FEColl(), 2, 1))); 
	else		
		fform.AddBdrFaceIntegrator(new ProjectedCoefBoundaryLFIntegrator(Jin2, *fes.FEColl(), 2, 1)); 
	fform.Assemble(); 
	Q0.Add(-1.0, fform); 
	Q1 *= 3.0; 		
}

IndependentBlockSMMOperator::IndependentBlockSMMOperator(
	mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &vfes, 
	const AngularQuadrature &quad, const MultiGroupEnergyGrid &energy, const TransportVectorExtents &psi_ext, 
	PhaseSpaceCoefficient &source_coef, PhaseSpaceCoefficient &inflow_coef, 
	double alpha, const BoundaryConditionMap &bc_map, int lumping)
	: fes(fes), vfes(vfes), quad(quad), psi_ext(psi_ext), source_coef(source_coef), inflow_coef(inflow_coef), 
	  alpha(alpha), bc_map(bc_map), lumping(lumping)
{ 
	width = TotalExtent(psi_ext);
	height = fes.GetVSize() + vfes.GetVSize();

	marshak_bdr_attrs = CreateBdrAttributeMarker<INFLOW>(bc_map);
	reflect_bdr_attrs = CreateBdrAttributeMarker<REFLECTIVE>(bc_map);

	Q0.SetSize(fes.GetVSize()); 
	Q0 = 0.0; 
	Q1.SetSize(vfes.GetVSize()); 
	Q1 = 0.0; 

	for (auto g=0; g<energy.Size(); g++) {
		source_coef.SetEnergy(energy.LowerBound(g), energy.UpperBound(g), energy.MeanEnergy(g));
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
	}

	InflowPartialCurrentCoefficient Jin_mg(inflow_coef, quad, energy); 
	VectorComponentSumCoefficient Jin(Jin_mg);
	mfem::ProductCoefficient Jin2(2.0, Jin); 
	mfem::LinearForm fform(&fes); 
	if (IsFaceLumped(lumping))
		fform.AddBdrFaceIntegrator(
			new QuadratureLumpedLFIntegrator(new ProjectedCoefBoundaryLFIntegrator(Jin2, *fes.FEColl(), 2, 1))); 
	else		
		fform.AddBdrFaceIntegrator(new ProjectedCoefBoundaryLFIntegrator(Jin2, *fes.FEColl(), 2, 1)); 
	fform.Assemble(); 
	Q0.Add(-1.0, fform); 
	Q1 *= 3.0; 		
}

void IndependentBlockSMMOperator::Mult(const mfem::Vector &psi, mfem::Vector &source) const 
{
	const bool lump_mass = IsMassLumped(lumping);
	const bool lump_grad = IsGradientLumped(lumping);
	const bool lump_face = IsFaceLumped(lumping);

	ConstTransportVectorView psi_view(psi.GetData(), psi_ext); 
	SMMCorrectionTensorCoefficient T(fes, quad, psi_view); 
	SMMBdrCorrectionFactorCoefficient beta(fes, quad, psi_view, alpha);

	mfem::ParLinearForm fform(&fes, source.GetData() + vfes.GetVSize()); 
	mfem::ProductCoefficient bdr_coef_f(-1.0, beta); 
	if (lump_face)
		fform.AddBdrFaceIntegrator(
			new QuadratureLumpedLFIntegrator(new ProjectedCoefBoundaryLFIntegrator(bdr_coef_f, *fes.FEColl(), 2, 1)), marshak_bdr_attrs); 
	else
		fform.AddBdrFaceIntegrator(new ProjectedCoefBoundaryLFIntegrator(bdr_coef_f, *fes.FEColl(), 2, 1), marshak_bdr_attrs); 
	fform.Assemble(); 

	mfem::ParLinearForm gform(&vfes, source.GetData()); 
	if (lump_grad)
		gform.AddDomainIntegrator(
			new QuadratureLumpedLFIntegrator(new WeakTensorDivergenceLFIntegrator(T))); 
	else
		gform.AddDomainIntegrator(new WeakTensorDivergenceLFIntegrator(T)); 
	if (lump_face) {
		gform.AddInteriorFaceIntegrator(
			new QuadratureLumpedLFIntegrator(new VectorJumpTensorAverageLFIntegrator(T))); 
		gform.AddBdrFaceIntegrator(
			new QuadratureLumpedLFIntegrator(new VectorJumpTensorAverageLFIntegrator(T))); 		
	} else {
		gform.AddInteriorFaceIntegrator(new VectorJumpTensorAverageLFIntegrator(T)); 
		gform.AddBdrFaceIntegrator(new VectorJumpTensorAverageLFIntegrator(T)); 
	}
	gform.Assemble();
	gform *= 3.0;  

	fform += Q0; 
	gform += Q1; 
}

IndependentRTSMMOperator::IndependentRTSMMOperator(
	mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &vfes, 
	const AngularQuadrature &quad, const TransportVectorExtents &psi_ext,
	const MultiGroupEnergyGrid &energy, PhaseSpaceCoefficient &source_coef, 
	PhaseSpaceCoefficient &inflow_coef, double alpha, 
	const BoundaryConditionMap &bc_map, int lumping)
	: fes(fes), vfes(vfes), quad(quad), psi_ext(psi_ext), alpha(alpha), 
	  bc_map(bc_map), lumping(lumping)
{
	width = TotalExtent(psi_ext);
	height = fes.GetVSize() + vfes.GetTrueVSize();

	offsets.SetSize(3);
	offsets[0] = 0;
	offsets[1] = vfes.GetTrueVSize();
	offsets[2] = fes.GetVSize();
	offsets.PartialSum();

	marshak_bdr_attrs = CreateBdrAttributeMarker<INFLOW>(bc_map);
	reflect_bdr_attrs = CreateBdrAttributeMarker<REFLECTIVE>(bc_map);
	vfes.GetEssentialTrueDofs(reflect_bdr_attrs, reflect_tdofs);

	Q0.SetSize(fes.GetVSize()); 
	Q0 = 0.0; 
	Q1.SetSize(vfes.GetVSize()); 
	Q1 = 0.0; 

	for (auto g=0; g<energy.Size(); g++) {
		source_coef.SetEnergy(energy.LowerBound(g), energy.UpperBound(g), energy.MeanEnergy(g));
		for (auto a=0; a<quad.Size(); a++) {
			const auto &Omega = quad.GetOmega(a); 
			source_coef.SetAngle(Omega); 
			mfem::LinearForm fform(&fes); 
			fform.AddDomainIntegrator(new mfem::DomainLFIntegrator(source_coef)); 
			fform.Assemble(); 

			mfem::VectorConstantCoefficient Omega_coef(Omega); 
			mfem::ScalarVectorProductCoefficient Omega_source(source_coef, Omega_coef);
			mfem::LinearForm gform(&vfes); 
			gform.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(Omega_source)); 
			gform.Assemble();  

			Q0.Add(quad.GetWeight(a), fform); 
			Q1.Add(quad.GetWeight(a), gform); 
		}		
	}

	InflowPartialCurrentCoefficient Jin_mg(inflow_coef, quad, energy); 
	VectorComponentSumCoefficient Jin(Jin_mg);
	mfem::ProductCoefficient bdr_coef(2.0/3.0/alpha, Jin); 
	mfem::LinearForm gform(&vfes); 
	mfem::LinearFormIntegrator *lfi = new VectorFEBoundaryNormalFaceLFIntegrator(bdr_coef, 2, 1);
	if (IsFaceLumped(lumping)) lfi = new QuadratureLumpedLFIntegrator(lfi);
	gform.AddBdrFaceIntegrator(lfi);
	gform.Assemble(); 
	Q1 += gform;
	Q1 *= 3.0;
}

void IndependentRTSMMOperator::Mult(const mfem::Vector &psi, mfem::Vector &source) const
{
	mfem::BlockVector block_source(source, offsets);
	const bool lump_mass = IsMassLumped(lumping);
	const bool lump_grad = IsGradientLumped(lumping);
	const bool lump_face = IsFaceLumped(lumping);

	ConstTransportVectorView psi_view(psi.GetData(), psi_ext); 
	SMMCorrectionTensorCoefficient T(fes, quad, psi_view); 
	MatrixDivergenceGridFunctionCoefficient divT(T);
	mfem::ScalarVectorProductCoefficient neg_divT(-1.0, divT);
	SMMBdrCorrectionFactorCoefficient beta(fes, quad, psi_view, alpha);
	mfem::ProductCoefficient bdr_coef(1.0/3.0/alpha, beta);

	mfem::ParLinearForm gform(&vfes);
	mfem::LinearFormIntegrator *lfi;
	// v . div(T) 
	lfi = new mfem::VectorFEDomainLFIntegrator(neg_divT);
	if (lump_grad) lfi = new QuadratureLumpedLFIntegrator(lfi);
	gform.AddDomainIntegrator(lfi);

	// - {v} . [Tn]
	lfi = new VectorFEAvgTensorJumpLFIntegrator(T);
	if (lump_face) lfi = new QuadratureLumpedLFIntegrator(lfi); 
	gform.AddInteriorFaceIntegrator(lfi);

	// v.n beta / 3 / alpha 
	lfi = new VectorFEBoundaryNormalFaceLFIntegrator(bdr_coef, 2, 1);
	if (lump_face) lfi = new QuadratureLumpedLFIntegrator(lfi);
	gform.AddBdrFaceIntegrator(lfi);
	gform.Assemble();
	gform *= 3.0;
	gform += Q1;

	gform.ParallelAssemble(block_source.GetBlock(0));
	block_source.GetBlock(0).SetSubVector(reflect_tdofs, 0.0);
	block_source.GetBlock(1) = Q0;
}