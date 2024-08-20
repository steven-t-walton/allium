#include "moment_discretization.hpp"
#include "lumping.hpp"
#include "linalg.hpp"
#include "moment_integrators.hpp"

MomentDiscretization::MomentDiscretization(
	mfem::ParFiniteElementSpace &fes, const BoundaryConditionMap &bc_map, int lumping)
	: fes(fes), bc_map(bc_map), lumping(lumping)
{
	marshak_bdr_attrs = CreateBdrAttributeMarker<INFLOW>(bc_map);
	reflect_bdr_attrs = CreateBdrAttributeMarker<REFLECTIVE>(bc_map);
}

void MomentDiscretization::SetTimeAbsorption(double sigma) 
{
	if (!Mtime) {
		const bool lump_mass = IsMassLumped(lumping);
		mfem::ParBilinearForm form(&fes); 
		if (lump_mass) 
			form.AddDomainIntegrator(new QuadratureLumpedIntegrator(new mfem::MassIntegrator));
		else
			form.AddDomainIntegrator(new mfem::MassIntegrator);
		form.Assemble(); 
		form.Finalize(); 
		Mtime.reset(form.ParallelAssemble()); 
	}
	time_absorption = sigma;
}

InteriorPenaltyDiscretization::InteriorPenaltyDiscretization(
	mfem::ParFiniteElementSpace &fes, const BoundaryConditionMap &bc_map, int lumping)
	: MomentDiscretization(fes, bc_map, lumping)
{
	SetKappa(kappa);
}

mfem::HypreParMatrix *InteriorPenaltyDiscretization::GetOperator(
	mfem::Coefficient &total, mfem::Coefficient &absorption) const 
{
	const bool lump_mass = IsMassLumped(lumping);
	const bool lump_grad = IsGradientLumped(lumping);
	const bool lump_face = IsFaceLumped(lumping);

	mfem::ConstantCoefficient alpha_coef(alpha/2);
	mfem::RatioCoefficient diffco(1.0/3, total);
	mfem::ParBilinearForm Kform(&fes); 
	mfem::BilinearFormIntegrator *bfi;
	bfi = new mfem::DiffusionIntegrator(diffco);
	if (lump_grad) bfi = new QuadratureLumpedIntegrator(bfi);
	Kform.AddDomainIntegrator(bfi);
	bfi = new mfem::MassIntegrator(absorption);
	if (lump_mass) bfi = new QuadratureLumpedIntegrator(bfi);
	Kform.AddDomainIntegrator(bfi);
	bfi = new MIPDiffusionIntegrator(diffco, sigma, kappa, mip_val);
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi);
	Kform.AddInteriorFaceIntegrator(bfi);
	bfi = new mfem::BoundaryMassIntegrator(alpha_coef);
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi);
	Kform.AddBdrFaceIntegrator(bfi, marshak_bdr_attrs);
	Kform.Assemble(); 
	Kform.Finalize(); 
	auto *K = Kform.ParallelAssemble();
	if (Mtime) K->Add(time_absorption, *Mtime);
	return K;
}

LDGDiscretization::LDGDiscretization(
	mfem::ParFiniteElementSpace &fes, const BoundaryConditionMap &bc_map, int lumping)
	: MomentDiscretization(fes, bc_map, lumping)
{
	auto *mesh = fes.GetParMesh();
	const auto dim = mesh->Dimension();
	vfes = std::make_unique<mfem::ParFiniteElementSpace>(mesh, fes.FEColl(), dim);
	beta.SetSize(dim);
	beta = 1.0;
}

mfem::HypreParMatrix *LDGDiscretization::GetOperator(
	mfem::Coefficient &total, mfem::Coefficient &absorption) const 
{
	const bool lump_mass = IsMassLumped(lumping);
	const bool lump_grad = IsGradientLumped(lumping);
	const bool lump_face = IsFaceLumped(lumping);

	mfem::ParBilinearForm Mtform(vfes.get());
	mfem::ProductCoefficient total3(3.0, total); 
	mfem::BilinearFormIntegrator *bfi;
	bfi = new mfem::VectorMassIntegrator(total3);
	if (lump_mass) bfi = new QuadratureLumpedIntegrator(bfi);
	bfi = new mfem::InverseIntegrator(bfi);
	Mtform.AddDomainIntegrator(bfi); 
	Mtform.Assemble(); 
	Mtform.Finalize();  
	auto iMt = HypreParMatrixPtr(Mtform.ParallelAssemble()); 

	mfem::ParBilinearForm Maform(&fes); 
	mfem::ConstantCoefficient alpha_c(alpha); 
	bfi = new mfem::MassIntegrator(absorption);
	if (lump_mass) bfi = new QuadratureLumpedIntegrator(bfi);
	Maform.AddDomainIntegrator(bfi); 
	bfi = new PenaltyIntegrator(alpha/2, false);
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi);
	Maform.AddInteriorFaceIntegrator(bfi); 
	bfi = new mfem::BoundaryMassIntegrator(alpha_c);
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi);
	Maform.AddBdrFaceIntegrator(bfi, marshak_bdr_attrs); 
	Maform.Assemble(); 
	Maform.Finalize(); 
	auto Ma = HypreParMatrixPtr(Maform.ParallelAssemble());

	mfem::ParMixedBilinearForm Dform(vfes.get(), &fes); 
	mfem::ConstantCoefficient neg_one(-1.0); 
	bfi = new mfem::TransposeIntegrator(new mfem::GradientIntegrator(neg_one));
	if (lump_grad) bfi = new QuadratureLumpedIntegrator(bfi);
	Dform.AddDomainIntegrator(bfi); 
	bfi = new mfem::LDGTraceIntegrator(&beta); 
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi);
	Dform.AddInteriorFaceIntegrator(bfi); 		
	Dform.Assemble(); 
	Dform.Finalize(); 
	auto D = HypreParMatrixPtr(Dform.ParallelAssemble()); 
	auto DT = HypreParMatrixPtr(D->Transpose()); 

	auto DiMt = HypreParMatrixPtr(mfem::ParMult(D.get(), iMt.get(), true)); 
	auto DiMtDT = HypreParMatrixPtr(mfem::ParMult(DiMt.get(), DT.get(), true)); 
	auto *S = mfem::ParAdd(DiMtDT.get(), Ma.get());
	if (Mtime) S->Add(time_absorption, *Mtime);
	return S;
}

BlockMomentDiscretization::BlockMomentDiscretization(
	mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &vfes, 
	const BoundaryConditionMap &bc_map, int lumping)
	: fes(fes), vfes(vfes), bc_map(bc_map), lumping(lumping)
{
	offsets.SetSize(3);
	offsets[0] = 0; 
	offsets[1] = vfes.GetVSize(); 
	offsets[2] = fes.GetVSize(); 
	offsets.PartialSum();

	marshak_bdr_attrs = CreateBdrAttributeMarker<INFLOW>(bc_map);
	reflect_bdr_attrs = CreateBdrAttributeMarker<REFLECTIVE>(bc_map);
}

void BlockMomentDiscretization::SetScalarTimeAbsorption(double sigma, const mfem::HypreParMatrix &M)
{
	time_absorption_s = sigma; 
	Mtime_s = &M;
}

void BlockMomentDiscretization::SetVectorTimeAbsorption(double sigma, const mfem::HypreParMatrix &M)
{
	time_absorption_v = sigma; 
	Mtime_v = &M;
}

void BlockMomentDiscretization::
Solver::SetOperator(const mfem::Operator &op) 
{
	const auto *ptr = dynamic_cast<const mfem::BlockOperator*>(&op);
	if (!ptr) MFEM_ABORT("operator must be a block operator");
	block_op = ptr;
	offsets = block_op->RowOffsets();

	height = width = block_op->Height();

	const auto *Mt = dynamic_cast<const mfem::HypreParMatrix*>(&block_op->GetBlock(0,0));
	const auto *DT = dynamic_cast<const mfem::HypreParMatrix*>(&block_op->GetBlock(0,1));
	const auto *D = dynamic_cast<const mfem::HypreParMatrix*>(&block_op->GetBlock(1,0));
	const auto *Ma = dynamic_cast<const mfem::HypreParMatrix*>(&block_op->GetBlock(1,1));

	iMt.reset(ElementByElementBlockInverse(vfes, *Mt));
	auto DiMt = HypreParMatrixPtr(mfem::ParMult(D, iMt.get(), true)); 
	auto DiMtDT = HypreParMatrixPtr(mfem::ParMult(DiMt.get(), DT, true)); 
	(*DiMtDT) *= -1.0;
	S.reset(mfem::ParAdd(DiMtDT.get(), Ma));
	schur_solver.SetOperator(*S);

	t1.SetSize(offsets[1] - offsets[0]);
	t2.SetSize(offsets[2] - offsets[1]);
}

void BlockMomentDiscretization::
Solver::Mult(const mfem::Vector &b, mfem::Vector &x) const
{
	mfem::Vector b1(*const_cast<mfem::Vector*>(&b), offsets[0], offsets[1] - offsets[0]);
	mfem::Vector b2(*const_cast<mfem::Vector*>(&b), offsets[1], offsets[2] - offsets[1]);
	mfem::Vector x1(x, offsets[0], offsets[1] - offsets[0]); 
	mfem::Vector x2(x, offsets[1] , offsets[2] - offsets[1]);

	iMt->Mult(b1, x1);
	block_op->GetBlock(1,0).Mult(x1, t2);
	add(b2, -1.0, t2, t2);
	schur_solver.Mult(t2, x2);
	block_op->GetBlock(0,1).Mult(x2, t1);
	add(b1, -1.0, t1, t1);
	iMt->Mult(t1, x1);
}

mfem::HypreParMatrix *BlockMomentDiscretization::FormSchurComplement(const mfem::Operator &op) const
{
	const auto *block_op = dynamic_cast<const mfem::BlockOperator*>(&op);
	if (!block_op) { MFEM_ABORT("operator must be a block operator"); }

	const auto *Mt = dynamic_cast<const mfem::HypreParMatrix*>(&block_op->GetBlock(0,0));
	const auto *DT = dynamic_cast<const mfem::HypreParMatrix*>(&block_op->GetBlock(0,1));
	const auto *D = dynamic_cast<const mfem::HypreParMatrix*>(&block_op->GetBlock(1,0));
	const auto *Ma = dynamic_cast<const mfem::HypreParMatrix*>(&block_op->GetBlock(1,1));

	auto iMt = HypreParMatrixPtr(ElementByElementBlockInverse(vfes, *Mt));
	auto DiMt = HypreParMatrixPtr(mfem::ParMult(D, iMt.get(), true)); 
	auto DiMtDT = HypreParMatrixPtr(mfem::ParMult(DiMt.get(), DT, true)); 
	(*DiMtDT) *= -1.0;
	auto *S = mfem::ParAdd(DiMtDT.get(), Ma);
	return S;
}

BlockLDGDiscretization::BlockLDGDiscretization(
	mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &vfes, 
	const BoundaryConditionMap &bc_map, int lumping)
	: BlockMomentDiscretization(fes, vfes, bc_map, lumping)
{
	const auto &mesh = *fes.GetMesh();
	beta.SetSize(mesh.Dimension());
	beta.Randomize(12345);
}

mfem::BlockOperator *BlockLDGDiscretization::GetOperator(mfem::Coefficient &total, mfem::Coefficient &absorption) const
{
	const bool lump_mass = IsMassLumped(lumping);
	const bool lump_grad = IsGradientLumped(lumping);
	const bool lump_face = IsFaceLumped(lumping);

	mfem::ParBilinearForm Mtform(&vfes);
	mfem::ProductCoefficient total3(3.0, total); 
	mfem::BilinearFormIntegrator *bfi;
	bfi = new mfem::VectorMassIntegrator(total3);
	if (lump_mass) bfi = new QuadratureLumpedIntegrator(bfi);
	Mtform.AddDomainIntegrator(bfi); 
	Mtform.Assemble(); 
	Mtform.Finalize();  
	auto *Mt = Mtform.ParallelAssemble(); 
	if (Mtime_v) Mt->Add(3.0*time_absorption_v, *Mtime_v);

	mfem::ParBilinearForm Maform(&fes); 
	mfem::ConstantCoefficient alpha_c(alpha); 
	bfi = new mfem::MassIntegrator(absorption);
	if (lump_mass) bfi = new QuadratureLumpedIntegrator(bfi);
	Maform.AddDomainIntegrator(bfi); 
	bfi = new PenaltyIntegrator(alpha/2, false);
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi);
	Maform.AddInteriorFaceIntegrator(bfi); 
	bfi = new mfem::BoundaryMassIntegrator(alpha_c);
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi);
	Maform.AddBdrFaceIntegrator(bfi, marshak_bdr_attrs); 
	Maform.Assemble(); 
	Maform.Finalize(); 
	auto *Ma = Maform.ParallelAssemble();
	if (Mtime_s) Ma->Add(time_absorption_s, *Mtime_s);

	mfem::ParMixedBilinearForm Dform(&vfes, &fes); 
	mfem::ConstantCoefficient neg_one(-1.0); 
	bfi = new mfem::TransposeIntegrator(new mfem::GradientIntegrator(neg_one));
	if (lump_grad) bfi = new QuadratureLumpedIntegrator(bfi);
	Dform.AddDomainIntegrator(bfi); 
	bfi = new mfem::LDGTraceIntegrator(&beta); 
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi);
	Dform.AddInteriorFaceIntegrator(bfi); 		
	Dform.Assemble(); 
	Dform.Finalize(); 
	auto *D = Dform.ParallelAssemble(); 
	auto *DT = D->Transpose(); 
	(*DT) *= -1.0;

	auto *op = new mfem::BlockOperator(offsets);
	op->SetBlock(0,0, Mt);
	op->SetBlock(0,1, DT);
	op->SetBlock(1,0, D);
	op->SetBlock(1,1, Ma);
	op->owns_blocks = 1;
	return op;
}

BlockIPDiscretization::BlockIPDiscretization(
	mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &vfes, 
	const BoundaryConditionMap &bc_map, int lumping)
	: BlockMomentDiscretization(fes, vfes, bc_map, lumping)
{
	SetKappa(kappa);
}

mfem::BlockOperator *BlockIPDiscretization::GetOperator(mfem::Coefficient &total, mfem::Coefficient &absorption) const
{
	const bool lump_mass = IsMassLumped(lumping);
	const bool lump_grad = IsGradientLumped(lumping);
	const bool lump_face = IsFaceLumped(lumping);

	mfem::ParBilinearForm Mtform(&vfes);
	mfem::ProductCoefficient total3(3.0, total); 
	mfem::BilinearFormIntegrator *bfi;
	bfi = new mfem::VectorMassIntegrator(total3);
	if (lump_mass) bfi = new QuadratureLumpedIntegrator(bfi);
	Mtform.AddDomainIntegrator(bfi); 
	Mtform.Assemble(); 
	Mtform.Finalize();  
	auto *Mt = Mtform.ParallelAssemble(); 
	if (Mtime_v) Mt->Add(3.0*time_absorption_v, *Mtime_v);

	mfem::ParBilinearForm Maform(&fes); 
	mfem::ConstantCoefficient alpha_c(alpha); 
	bfi = new mfem::MassIntegrator(absorption);
	if (lump_mass) bfi = new QuadratureLumpedIntegrator(bfi);
	Maform.AddDomainIntegrator(bfi); 
	mfem::RatioCoefficient diffco(1.0/3, total);
	mfem::Coefficient *coef = (scale_penalty) ? &diffco : nullptr;
	bfi = new PenaltyIntegrator(kappa, mip_val, coef);
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi);
	Maform.AddInteriorFaceIntegrator(bfi); 
	bfi = new mfem::BoundaryMassIntegrator(alpha_c);
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi);
	Maform.AddBdrFaceIntegrator(bfi, marshak_bdr_attrs); 
	Maform.Assemble(); 
	Maform.Finalize(); 
	auto *Ma = Maform.ParallelAssemble();
	if (Mtime_s) Ma->Add(time_absorption_s, *Mtime_s);

	mfem::ParMixedBilinearForm Dform(&vfes, &fes); 
	mfem::ConstantCoefficient neg_one(-1.0); 
	bfi = new mfem::TransposeIntegrator(new mfem::GradientIntegrator(neg_one));
	if (lump_grad) bfi = new QuadratureLumpedIntegrator(bfi);
	Dform.AddDomainIntegrator(bfi); 
	bfi = new DGJumpAverageIntegrator; 
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi);
	Dform.AddInteriorFaceIntegrator(bfi); 		
	Dform.Assemble(); 
	Dform.Finalize(); 
	auto *D = Dform.ParallelAssemble(); 
	auto *DT = D->Transpose(); 
	(*DT) *= -1.0;

	auto *op = new mfem::BlockOperator(offsets);
	op->SetBlock(0,0, Mt);
	op->SetBlock(0,1, DT);
	op->SetBlock(1,0, D);
	op->SetBlock(1,1, Ma);
	op->owns_blocks = 1;
	return op;
}

P1Discretization::P1Discretization(
	mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &vfes, 
	const BoundaryConditionMap &bc_map, int lumping)
	: BlockMomentDiscretization(fes, vfes, bc_map, lumping)
{
}

mfem::BlockOperator *P1Discretization::GetOperator(mfem::Coefficient &total, mfem::Coefficient &absorption) const
{
	const bool lump_mass = IsMassLumped(lumping);
	const bool lump_grad = IsGradientLumped(lumping);
	const bool lump_face = IsFaceLumped(lumping);

	mfem::ParBilinearForm Mtform(&vfes);
	mfem::ProductCoefficient total3(3.0, total); 
	if (lump_mass) {
		Mtform.AddDomainIntegrator(new QuadratureLumpedIntegrator(new mfem::VectorMassIntegrator(total3)));
	} else {
		Mtform.AddDomainIntegrator(new mfem::VectorMassIntegrator(total3));
	}
	if (lump_face) {
		Mtform.AddInteriorFaceIntegrator(new QuadratureLumpedIntegrator(new DGVectorJumpJumpIntegrator(1.0/2/alpha))); 
		Mtform.AddBdrFaceIntegrator(new QuadratureLumpedIntegrator(new DGVectorJumpJumpIntegrator(1.0/2/alpha)), marshak_bdr_attrs);
		Mtform.AddBdrFaceIntegrator(new QuadratureLumpedIntegrator(new DGVectorJumpJumpIntegrator(1.0/alpha)), reflect_bdr_attrs);  		
	} else {
		Mtform.AddInteriorFaceIntegrator(new DGVectorJumpJumpIntegrator(1.0/2/alpha)); 
		Mtform.AddBdrFaceIntegrator(new DGVectorJumpJumpIntegrator(1.0/2/alpha), marshak_bdr_attrs);
		Mtform.AddBdrFaceIntegrator(new DGVectorJumpJumpIntegrator(1.0/alpha), reflect_bdr_attrs);  
	}
	Mtform.Assemble(); 
	Mtform.Finalize();  
	mfem::HypreParMatrix *Mt = Mtform.ParallelAssemble(); 
	if (Mtime_v) Mt->Add(3.0*time_absorption_v, *Mtime_v);

	mfem::ParBilinearForm Maform(&fes); 
	mfem::ConstantCoefficient alpha_c(alpha/2); 
	if (lump_mass) {
		Maform.AddDomainIntegrator(new QuadratureLumpedIntegrator(new mfem::MassIntegrator(absorption))); 
	} else {
		Maform.AddDomainIntegrator(new mfem::MassIntegrator(absorption)); 
	}
	if (lump_face) {
		Maform.AddInteriorFaceIntegrator(new QuadratureLumpedIntegrator(new PenaltyIntegrator(alpha/2, false))); 
		Maform.AddBdrFaceIntegrator(new QuadratureLumpedIntegrator(new mfem::BoundaryMassIntegrator(alpha_c)), marshak_bdr_attrs); 		
	} else {
		Maform.AddInteriorFaceIntegrator(new PenaltyIntegrator(alpha/2, false)); 
		Maform.AddBdrFaceIntegrator(new mfem::BoundaryMassIntegrator(alpha_c), marshak_bdr_attrs); 
	}
	Maform.Assemble(); 
	Maform.Finalize(); 
	mfem::HypreParMatrix *Ma = Maform.ParallelAssemble();
	if (Mtime_s) Ma->Add(time_absorption_s, *Mtime_s);

	mfem::ParMixedBilinearForm Dform(&vfes, &fes); 
	mfem::ConstantCoefficient neg_one(-1.0); 
	if (lump_grad) {
		Dform.AddDomainIntegrator(new QuadratureLumpedIntegrator(new mfem::TransposeIntegrator(new mfem::GradientIntegrator(neg_one)))); 		
	} else {
		Dform.AddDomainIntegrator(new mfem::TransposeIntegrator(new mfem::GradientIntegrator(neg_one))); 
	}
	if (lump_face) {
		Dform.AddInteriorFaceIntegrator(new QuadratureLumpedIntegrator(new DGJumpAverageIntegrator)); 
		Dform.AddBdrFaceIntegrator(new QuadratureLumpedIntegrator(new DGJumpAverageIntegrator(0.5)), marshak_bdr_attrs); 		
	} else {
		Dform.AddInteriorFaceIntegrator(new DGJumpAverageIntegrator); 
		Dform.AddBdrFaceIntegrator(new DGJumpAverageIntegrator(0.5), marshak_bdr_attrs); 
	}
	Dform.Assemble(); 
	Dform.Finalize(); 
	mfem::HypreParMatrix *D = Dform.ParallelAssemble(); 
	mfem::HypreParMatrix *G = D->Transpose(); 
	(*G) *= -1.0; 

	// mfem::ParMixedBilinearForm Gform(&fes, &vfes); 
	// mfem::ConstantCoefficient pos_one(1.0); 
	// Gform.AddDomainIntegrator(new mfem::GradientIntegrator(pos_one)); 
	// Gform.AddInteriorFaceIntegrator(new mfem::TransposeIntegrator(new DGJumpAverageIntegrator(-1.0))); 
	// Gform.AddBdrFaceIntegrator(new mfem::TransposeIntegrator(new DGJumpAverageIntegrator(-0.5)), marshak_bdr_attrs); 
	// Gform.Assemble(); 
	// Gform.Finalize(); 
	// mfem::HypreParMatrix *G = Gform.ParallelAssemble(); 

	auto *op = new mfem::BlockOperator(offsets);
	op->SetBlock(0,0, Mt); 
	op->SetBlock(0,1, G); 
	op->SetBlock(1,0, D); 
	op->SetBlock(1,1, Ma); 
	op->owns_blocks = 1;
	return op;
}