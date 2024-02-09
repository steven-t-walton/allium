#include "cons_smm_op.hpp"
#include "smm_integrators.hpp"
#include "transport_op.hpp"
#include "p1diffusion.hpp"
#include "mip.hpp"

ConsistentSMMSourceOperator::ConsistentSMMSourceOperator(mfem::ParFiniteElementSpace &_fes, mfem::ParFiniteElementSpace &_vfes, 
		const AngularQuadrature &_quad, const TransportVectorExtents &_psi_ext, ConstTransportVectorView source_vec, double _alpha)
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
	Q1 *= 3.0; 

	auto face_to_element = std::unique_ptr<const mfem::Table>(fes.GetParMesh()->GetFaceToAllElementTable()); 
	element_to_face = std::unique_ptr<const mfem::Table>(mfem::Transpose(*face_to_element)); 

	if (fes.IsVariableOrder()) { MFEM_ABORT("variable order not supported for consistent SMM"); }
	trace_coll = std::make_unique<DGTrace_FECollection>(fes.GetOrder(0), dim); 
	auto &mesh = *fes.GetParMesh(); 
	trace_fes = std::make_unique<mfem::ParFiniteElementSpace>(&mesh, trace_coll.get()); 
	trace_vfes = std::make_unique<mfem::ParFiniteElementSpace>(&mesh, trace_coll.get(), dim); 
	beta.SetSpace(trace_fes.get());
	tensor.SetSpace(trace_vfes.get());  
}

void ConsistentSMMSourceOperator::Mult(const mfem::Vector &psi, mfem::Vector &source) const 
{
	mfem::BlockVector bv(source.GetData(), offsets); 
	ConstTransportVectorView psi_view(psi.GetData(), psi_ext); 
	SMMCorrectionTensorCoefficient T(fes, quad, psi_view);
	ProjectClosuresToFaces(fes, quad, psi_view, alpha, beta, tensor); 

	mfem::ParLinearForm fform(&fes, bv.GetBlock(1).GetData()); 
	fform.AddInteriorFaceIntegrator(new CSMMZerothMomentFaceLFIntegrator(beta));
	fform.AddBdrFaceIntegrator(new CSMMZerothMomentFaceLFIntegrator(beta)); 
	// fform.AddInteriorFaceIntegrator(new CSMMFaceIntegrator0(fes, quad, psi_view, alpha)); 
	// fform.AddBdrFaceIntegrator(new CSMMFaceIntegrator0(fes, quad, psi_view, alpha)); 
	fform.Assemble(); 

	mfem::ParLinearForm gform(&vfes, bv.GetBlock(0).GetData()); 
	gform.AddDomainIntegrator(new WeakTensorDivergenceLFIntegrator(T)); 
	gform.AddInteriorFaceIntegrator(new CSMMFirstMomentFaceLFIntegrator(tensor)); 
	gform.AddBdrFaceIntegrator(new CSMMFirstMomentFaceLFIntegrator(tensor)); 
	// gform.AddBdrFaceIntegrator(new CSMMFaceIntegrator1(fes, quad, psi_view, alpha)); 
	// gform.AddInteriorFaceIntegrator(new CSMMFaceIntegrator1(fes, quad, psi_view, alpha)); 
	gform.Assemble(); 
	gform *= 3.0; 

	fform += Q0; 
	gform += Q1; 
}

ConsistentLDGSMMSourceOperator::ConsistentLDGSMMSourceOperator(mfem::ParFiniteElementSpace &_fes, mfem::ParFiniteElementSpace &_vfes, 
	const AngularQuadrature &_quad, const TransportVectorExtents &_psi_ext, ConstTransportVectorView source_vec, 
	double _alpha, const mfem::Vector &beta, mfem::Coefficient *total)
	: fes(_fes), vfes(_vfes), quad(_quad), psi_ext(_psi_ext), alpha(_alpha), 
	  base_source_op(_fes, _vfes, _quad, _psi_ext, source_vec, _alpha)
{
	height = base_source_op.Height(); 
	width = base_source_op.Width(); 

	const auto dim = quad.Dimension(); 
	phi_ext = MomentVectorExtents(1,dim+1,fes.GetVSize()); 
	moments.SetSize(TotalExtent(phi_ext)); 

	mfem::ParBilinearForm F1form(&vfes); 
	F1form.AddInteriorFaceIntegrator(new DGVectorJumpJumpIntegrator(-1.0/2/alpha)); 
	F1form.Assemble(); 
	F1form.Finalize(); 
	F1 = HypreParMatrixPtr(F1form.ParallelAssemble()); 

	mfem::ParMixedBilinearForm F2form(&vfes, &fes); 
	F2form.AddInteriorFaceIntegrator(new DGJumpAverageIntegrator(-1.0));
	mfem::Coefficient *diffco = nullptr; 
	if (total) {
		diffco = new mfem::RatioCoefficient(1./3, *total); 
		F2form.AddInteriorFaceIntegrator(new LDGTraceIntegrator(*diffco, &beta)); 
	} else {
		F2form.AddInteriorFaceIntegrator(new mfem::LDGTraceIntegrator(&beta));		
	}
	F2form.Assemble(); 
	F2form.Finalize(); 
	F2 = HypreParMatrixPtr(F2form.ParallelAssemble());   

	if (diffco) delete diffco; 
}

void ConsistentLDGSMMSourceOperator::Mult(const mfem::Vector &psi, mfem::Vector &source) const 
{
	base_source_op.Mult(psi, source); 

	mfem::BlockVector bv(source.GetData(), base_source_op.Offsets()); 

	const auto dim = quad.Dimension(); 
	DiscreteToMoment D(quad, psi_ext, phi_ext); 
	D.Mult(psi, moments); 
	mfem::Vector phi(moments, 0, fes.GetVSize()); 
	mfem::Vector J(moments, fes.GetVSize(), dim*fes.GetVSize()); 
	F1->Mult(1.0, J, 1.0, bv.GetBlock(0)); 
	F2->Mult(1.0, J, 1.0, bv.GetBlock(1)); 
	F2->MultTranspose(-1.0, phi, 1.0, bv.GetBlock(0)); 
}

ConsistentIPSMMSourceOperator::ConsistentIPSMMSourceOperator(mfem::ParFiniteElementSpace &_fes, mfem::ParFiniteElementSpace &_vfes, 
	const AngularQuadrature &_quad, const TransportVectorExtents &_psi_ext, ConstTransportVectorView source_vec, 
	double _alpha, mfem::Coefficient &total, double _kappa, bool mip, bool scale_ip_stabilization)
	: fes(_fes), vfes(_vfes), quad(_quad), psi_ext(_psi_ext), alpha(_alpha), kappa(_kappa),
	  base_source_op(_fes, _vfes, _quad, _psi_ext, source_vec, _alpha)
{
	height = base_source_op.Height(); 
	width = base_source_op.Width(); 

	if (kappa < 0.0) {
		kappa *= -1.0 * pow(fes.GetOrder(0)+1, 2); 
	} 

	const auto dim = quad.Dimension(); 
	phi_ext = MomentVectorExtents(1,dim+1,fes.GetVSize()); 
	moments.SetSize(TotalExtent(phi_ext)); 

	mfem::ParBilinearForm F1form(&vfes); 
	F1form.AddInteriorFaceIntegrator(new DGVectorJumpJumpIntegrator(-1.0/2/alpha)); 
	F1form.Assemble(); 
	F1form.Finalize(); 
	F1 = HypreParMatrixPtr(F1form.ParallelAssemble()); 

	mfem::ParBilinearForm F2form(&fes); 
	F2form.AddInteriorFaceIntegrator(new PenaltyIntegrator(-alpha/2, false)); 
	mfem::RatioCoefficient diffco(1./3, total); 
	mfem::Coefficient *coef = scale_ip_stabilization ? &diffco : nullptr; 
	double limit = mip ? alpha/2 : 0.0; 
	F2form.AddInteriorFaceIntegrator(new PenaltyIntegrator(kappa, limit, coef)); 
	F2form.Assemble(); 
	F2form.Finalize(); 
	F2 = HypreParMatrixPtr(F2form.ParallelAssemble());   
}

void ConsistentIPSMMSourceOperator::Mult(const mfem::Vector &psi, mfem::Vector &source) const 
{
	base_source_op.Mult(psi, source); 

	mfem::BlockVector bv(source.GetData(), base_source_op.Offsets()); 

	const auto dim = quad.Dimension(); 
	DiscreteToMoment D(quad, psi_ext, phi_ext); 
	D.Mult(psi, moments); 
	mfem::Vector phi(moments, 0, fes.GetVSize()); 
	mfem::Vector J(moments, fes.GetVSize(), dim*fes.GetVSize()); 
	F1->Mult(1.0, J, 1.0, bv.GetBlock(0)); 
	F2->Mult(1.0, phi, 1.0, bv.GetBlock(1)); 
}

void ProjectClosuresToFaces(const mfem::ParFiniteElementSpace &fes, const AngularQuadrature &quad, ConstTransportVectorView psi, 
	double alpha, mfem::ParGridFunction &beta, mfem::ParGridFunction &tensor)
{
	auto &mesh = *fes.GetParMesh(); 
	std::unique_ptr<const mfem::Table> face_to_element(mesh.GetFaceToAllElementTable()); 
	std::unique_ptr<const mfem::Table> element_to_face(mfem::Transpose(*face_to_element));  
	const auto dim = mesh.Dimension(); 
	const auto &beta_fes = *beta.FESpace(); 
	const auto &tensor_fes = *tensor.FESpace(); 
	mfem::Vector nor(dim), shape, psi_local; 
	mfem::Array<int> psi_dof, beta_dof, tensor_dof; 
	for (auto e=0; e<fes.GetNE(); e++) {
		const auto &psi_el = *fes.GetFE(e); 
		shape.SetSize(psi_el.GetDof()); 
		psi_local.SetSize(psi_el.GetDof()); 
		fes.GetElementDofs(e, psi_dof);
		beta_fes.GetElementDofs(e, beta_dof);  
		tensor_fes.GetElementVDofs(e, tensor_dof); 
		const auto *faces = element_to_face->GetRow(e); 
		for (auto f=0; f<element_to_face->RowSize(e); f++) {
			const auto face = faces[f]; 
			const auto &info = mesh.GetFaceInformation(face); 
			mfem::FaceElementTransformations *trans; 
			if (info.IsShared()) {
				trans = mesh.GetSharedFaceTransformationsByLocalIndex(face, true); 
			} else {
				trans = mesh.GetFaceElementTransformations(face); 
			}			
			bool keep_order = e == trans->Elem1No; 
			const auto local_face_id = info.element[keep_order ? 0 : 1].local_face_id; 
			const auto &tr_el = *beta_fes.GetTraceElement(face, mesh.GetFaceGeometry(face)); 
			const auto &ir = tr_el.GetNodes(); 

			mfem::IntegrationPoint eip; 
			for (auto n=0; n<ir.GetNPoints(); n++) {
				const auto &ip = ir.IntPoint(n); 
				trans->SetAllIntPoints(&ip); 
				if (dim==1) {
					nor(0) = 2*trans->GetElement1IntPoint().x - 1.0;
				} else {
					mfem::CalcOrtho(trans->Jacobian(), nor); 				
				}
				nor.Set((keep_order ? 1.0 : -1.0)/nor.Norml2(), nor); 

				if (keep_order) {
					eip = trans->GetElement1IntPoint(); 
				} else {
					eip = trans->GetElement2IntPoint(); 
				}

				psi_el.CalcShape(eip, shape); 

				double val = 0.0, phi = 0.0, Jn = 0.0; 
				mfem::Vector ten_val(dim); 
				ten_val = 0.0; 
				for (auto a=0; a<quad.Size(); a++) {
					const auto &Omega = quad.GetOmega(a); 
					double w = quad.GetWeight(a); 
					double dot = Omega * nor; 
					for (auto i=0; i<psi_dof.Size(); i++) { psi_local[i] = psi(0,a,psi_dof[i]); }
					double psi_at_ip = shape * psi_local; 
					val += (std::fabs(Omega * nor) - alpha) * psi_at_ip * w; 
					phi += psi_at_ip * w; 
					Jn += dot * psi_at_ip * w; 
					if (dot > 0) {
						for (auto d=0; d<dim; d++) {
							ten_val(d) += Omega(d) * dot * psi_at_ip * w; 
						}
					}
				}
				beta(beta_dof[local_face_id*tr_el.GetDof() + n]) = val; 
				for (auto d=0; d<dim; d++) {
					tensor(tensor_dof[local_face_id*tr_el.GetDof()*dim + n + d*tr_el.GetDof()]) 
						= ten_val(d) - nor(d)/6*phi - nor(d)/6/alpha*Jn; 
				}
			}
		}
	}

	beta.ExchangeFaceNbrData(); 
	tensor.ExchangeFaceNbrData(); 
}