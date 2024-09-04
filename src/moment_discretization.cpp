#include "moment_discretization.hpp"
#include "lumping.hpp"
#include "linalg.hpp"
#include "moment_integrators.hpp"

MomentDiscretization::MomentDiscretization(
	mfem::ParFiniteElementSpace &fes, mfem::Coefficient &total, mfem::Coefficient &absorption, 
	const BoundaryConditionMap &bc_map, int lumping)
	: fes(fes), total(total), absorption(absorption), bc_map(bc_map), lumping(lumping)
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

H1DiffusionDiscretization::H1DiffusionDiscretization(
	mfem::ParFiniteElementSpace &fes, mfem::Coefficient &total, 
	mfem::Coefficient &absorption, const BoundaryConditionMap &bc_map, 
	int lumping)
	: MomentDiscretization(fes, total, absorption, bc_map, lumping)
{ }

mfem::HypreParMatrix *H1DiffusionDiscretization::GetOperator() const
{
	const bool lump_mass = IsMassLumped(lumping);
	const bool lump_grad = IsGradientLumped(lumping);
	const bool lump_face = IsFaceLumped(lumping);

	mfem::ConstantCoefficient alpha_coef(alpha);
	mfem::RatioCoefficient diffco(1.0/3, total);
	mfem::ParBilinearForm Kform(&fes); 
	mfem::BilinearFormIntegrator *bfi;
	bfi = new mfem::DiffusionIntegrator(diffco);
	if (lump_grad) bfi = new QuadratureLumpedIntegrator(bfi);
	Kform.AddDomainIntegrator(bfi);
	bfi = new mfem::MassIntegrator(absorption);
	if (lump_mass) bfi = new QuadratureLumpedIntegrator(bfi);
	Kform.AddDomainIntegrator(bfi);
	bfi = new mfem::BoundaryMassIntegrator(alpha_coef);
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi);
	Kform.AddBdrFaceIntegrator(bfi, marshak_bdr_attrs);
	Kform.Assemble();
	Kform.Finalize();
	auto *K = Kform.ParallelAssemble();
	if (Mtime) K->Add(time_absorption, *Mtime);
	return K;
}

InteriorPenaltyDiscretization::InteriorPenaltyDiscretization(
	mfem::ParFiniteElementSpace &fes, mfem::Coefficient &total, 
	mfem::Coefficient &absorption, const BoundaryConditionMap &bc_map, 
	int lumping)
	: MomentDiscretization(fes, total, absorption, bc_map, lumping)
{
	SetKappa(kappa);
}

mfem::HypreParMatrix *InteriorPenaltyDiscretization::GetOperator() const 
{
	const bool lump_mass = IsMassLumped(lumping);
	const bool lump_grad = IsGradientLumped(lumping);
	const bool lump_face = IsFaceLumped(lumping);

	// total is required across parallel faces -> exchange 
	// grid function data if coefficient is a grid function coefficient 
	auto *total_ptr = dynamic_cast<mfem::GridFunctionCoefficient*>(&total);
	if (total_ptr) {
		const auto *pgf = dynamic_cast<const mfem::ParGridFunction*>(total_ptr->GetGridFunction());
		if (pgf)
			const_cast<mfem::ParGridFunction*>(pgf)->ExchangeFaceNbrData();
	}

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
	mfem::ParFiniteElementSpace &fes, 
	mfem::Coefficient &total, mfem::Coefficient &absorption, 
	const BoundaryConditionMap &bc_map, int lumping)
	: MomentDiscretization(fes, total, absorption, bc_map, lumping)
{
	auto *mesh = fes.GetParMesh();
	const auto dim = mesh->Dimension();
	vfes = std::make_unique<mfem::ParFiniteElementSpace>(mesh, fes.FEColl(), dim);
	beta.SetSize(dim);
	beta = 1.0;
}

mfem::HypreParMatrix *LDGDiscretization::GetOperator() const 
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
	mfem::Coefficient &total, mfem::Coefficient &absorption,
	const BoundaryConditionMap &bc_map, int lumping)
	: fes(fes), vfes(vfes), total(total), absorption(absorption), 
	  bc_map(bc_map), lumping(lumping)
{
	offsets.SetSize(3);
	offsets[0] = 0; 
	offsets[1] = vfes.GetTrueVSize(); 
	offsets[2] = fes.GetTrueVSize(); 
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
	mfem::Coefficient &total, mfem::Coefficient &absorption,
	const BoundaryConditionMap &bc_map, int lumping)
	: BlockMomentDiscretization(fes, vfes, total, absorption, bc_map, lumping)
{
	const auto &mesh = *fes.GetMesh();
	beta.SetSize(mesh.Dimension());
	beta.Randomize(12345);
}

mfem::BlockOperator *BlockLDGDiscretization::GetOperator() const
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
	mfem::Coefficient &total, mfem::Coefficient &absorption,
	const BoundaryConditionMap &bc_map, int lumping)
	: BlockMomentDiscretization(fes, vfes, total, absorption, bc_map, lumping)
{
	SetKappa(kappa);
}

mfem::BlockOperator *BlockIPDiscretization::GetOperator() const
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
	mfem::Coefficient &total, mfem::Coefficient &absorption,
	const BoundaryConditionMap &bc_map, int lumping)
	: BlockMomentDiscretization(fes, vfes, total, absorption, bc_map, lumping)
{
}

mfem::BlockOperator *P1Discretization::GetOperator() const
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

RTDiffusionDiscretization::RTDiffusionDiscretization(
	mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &vfes, 
	mfem::Coefficient &total, mfem::Coefficient &absorption,
	const BoundaryConditionMap &bc_map, int lumping)
	: BlockMomentDiscretization(fes, vfes, total, absorption, bc_map, lumping)
{
	vfes.GetEssentialTrueDofs(reflect_bdr_attrs, ess_tdof_list);
}

mfem::BlockOperator *RTDiffusionDiscretization::GetOperator() const
{
	const bool lump_mass = IsMassLumped(lumping);
	const bool lump_grad = IsGradientLumped(lumping);
	const bool lump_face = IsFaceLumped(lumping);

	mfem::BilinearFormIntegrator *bfi;

	mfem::ParBilinearForm Mtform(&vfes);
	mfem::ProductCoefficient total3(3.0, total);
	mfem::ConstantCoefficient marshak_coef(1.0/alpha);
	bfi = new mfem::VectorFEMassIntegrator(total3);
	if (lump_mass) bfi = new QuadratureLumpedIntegrator(bfi);
	Mtform.AddDomainIntegrator(bfi);
	bfi = new mfem::BoundaryMassIntegrator(marshak_coef);
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi);
	Mtform.AddBoundaryIntegrator(bfi, marshak_bdr_attrs);
	Mtform.Assemble();
	Mtform.Finalize();
	auto *Mt = Mtform.ParallelAssemble();
	delete Mt->EliminateRowsCols(ess_tdof_list);

	mfem::ParMixedBilinearForm Dform(&vfes, &fes);
	bfi = new mfem::VectorFEDivergenceIntegrator;
	if (lump_grad) bfi = new QuadratureLumpedIntegrator(bfi);
	Dform.AddDomainIntegrator(bfi);
	Dform.Assemble();
	Dform.Finalize();
	auto *D = Dform.ParallelAssemble();
	delete D->EliminateCols(ess_tdof_list);
	auto *G = D->Transpose();
	(*G) *= -1.0;

	mfem::ParBilinearForm Maform(&fes);
	bfi = new mfem::MassIntegrator(absorption);
	if (lump_mass) bfi = new QuadratureLumpedIntegrator(bfi);
	Maform.AddDomainIntegrator(bfi);
	Maform.Assemble();
	Maform.Finalize();
	auto *Ma = Maform.ParallelAssemble();
	if (Mtime_s) Ma->Add(time_absorption_s, *Mtime_s);

	auto *op = new mfem::BlockOperator(offsets);
	op->SetBlock(0,0, Mt); 
	op->SetBlock(0,1, G); 
	op->SetBlock(1,0, D); 
	op->SetBlock(1,1, Ma); 
	op->owns_blocks = 1;
	return op;
}

HybridizedRTDiffusionDiscretization::HybridizedRTDiffusionDiscretization(
	mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &vfes, 
	mfem::ParFiniteElementSpace &ifes, mfem::Coefficient &total, 
	mfem::Coefficient &absorption, const BoundaryConditionMap &bc_map, 
	int lumping)
	: ifes(ifes), BlockMomentDiscretization(fes, vfes, total, absorption, bc_map, lumping)
{
	rt_br_dofs.MakeI(fes.GetNE());
	mfem::Array<int> vdofs, dofs;
	for (int e=0; e<fes.GetNE(); e++) {
		vfes.GetElementVDofs(e, vdofs);
		rt_br_dofs.AddColumnsInRow(e, vdofs.Size());
	}
	rt_br_dofs.MakeJ();
	int dof_count = 0;
	for (int e=0; e<fes.GetNE(); e++) {
		vfes.GetElementVDofs(e, vdofs);
		for (int i=0; i<vdofs.Size(); i++) {
			rt_br_dofs.AddConnection(e, dof_count++);
		}
	}
	rt_br_dofs.ShiftUpI();

	mfem::Array<int> ess_tdof_list;
	vfes.GetEssentialTrueDofs(reflect_bdr_attrs, ess_tdof_list);
	mfem::Array<int> tdof_marker_local(vfes.GetTrueVSize());
	tdof_marker_local = 1;
	for (int i=0; i<ess_tdof_list.Size(); i++) {
		tdof_marker_local[ess_tdof_list[i]] = 0;
	}

	mfem::Array<int> tdof_marker(vfes.GetVSize());
	vfes.Dof_TrueDof_Matrix()->BooleanMult(1, tdof_marker_local, 0, tdof_marker);

	mfem::Array<int> br_vdofs;
	br_tdof_marker.SetSize(dof_count);
	br_tdof_marker = 0;
	for (int e=0; e<fes.GetNE(); e++) {
		vfes.GetElementVDofs(e, vdofs);
		mfem::FiniteElementSpace::AdjustVDofs(vdofs);
		rt_br_dofs.GetRow(e, br_vdofs);
		for (int i=0; i<vdofs.Size(); i++) {
			br_tdof_marker[br_vdofs[i]] = !tdof_marker[vdofs[i]];
		}
	}

	offsets.SetSize(4);
	offsets[0] = 0;
	offsets[1] = dof_count;
	offsets[2] = fes.GetVSize();
	offsets[3] = ifes.GetVSize();
	offsets.PartialSum();
}

mfem::BlockOperator *HybridizedRTDiffusionDiscretization::GetOperator() const
{
	const bool lump_mass = IsMassLumped(lumping);
	const bool lump_grad = IsGradientLumped(lumping);
	const bool lump_face = IsFaceLumped(lumping);

	mfem::BilinearFormIntegrator *bfi;

	mfem::ParBilinearForm Mtform(&vfes);
	mfem::ProductCoefficient total3(3.0, total);
	mfem::ConstantCoefficient marshak_coef(1.0/alpha);
	bfi = new mfem::VectorFEMassIntegrator(total3);
	if (lump_mass) bfi = new QuadratureLumpedIntegrator(bfi);
	Mtform.AddDomainIntegrator(bfi);
	bfi = new mfem::BoundaryMassIntegrator(marshak_coef);
	if (lump_face) bfi = new QuadratureLumpedIntegrator(bfi);
	Mtform.AddBoundaryIntegrator(bfi, marshak_bdr_attrs);

	mfem::ParMixedBilinearForm Dform(&vfes, &fes);
	bfi = new mfem::VectorFEDivergenceIntegrator;
	if (lump_grad) bfi = new QuadratureLumpedIntegrator(bfi);
	Dform.AddDomainIntegrator(bfi);

	mfem::ParBilinearForm Maform(&fes);
	bfi = new mfem::MassIntegrator(absorption);
	if (lump_mass) bfi = new QuadratureLumpedIntegrator(bfi);
	Maform.AddDomainIntegrator(bfi);

	const auto &dg_dofs = fes.GetElementToDofTable();

	auto *Mt = new DenseBlockDiagonalOperator(rt_br_dofs, rt_br_dofs);
	auto *D = new DenseBlockDiagonalOperator(dg_dofs, rt_br_dofs);
	auto *G = new DenseBlockDiagonalOperator(rt_br_dofs, dg_dofs);
	auto *Ma = new DenseBlockDiagonalOperator(dg_dofs, dg_dofs);

	mfem::DenseMatrix elmat;
	for (int e=0; e<fes.GetNE(); e++) {
		Mtform.ComputeElementMatrix(e, elmat);
		Mt->SetBlock(e, elmat);		
	}

	auto &mesh = *vfes.GetParMesh();
	for (int be=0; be<fes.GetNBE(); be++) {
		Mtform.ComputeBdrElementMatrix(be, elmat);
		int el, info, vdim = vfes.GetVDim();
		mesh.GetBdrElementAdjacentElement(be, el, info);
		mfem::Array<int> e2f(rt_br_dofs.RowSize(el)), lvdofs(elmat.Height());
		e2f = -1;
		vfes.FEColl()->SubDofOrder(mesh.GetElementBaseGeometry(el), mesh.Dimension()-1, info, lvdofs);
		mfem::Ordering::DofsToVDofs<mfem::Ordering::byNODES>(e2f.Size()/vdim, vdim, lvdofs);
		elmat.AdjustDofDirection(lvdofs);
		mfem::FiniteElementSpace::AdjustVDofs(lvdofs);
		for (int i=0; i<lvdofs.Size(); i++) {
			e2f[lvdofs[i]] = i;
		}
		auto &mt = Mt->GetBlock(el);
		for (int i=0; i<mt.Height(); i++) {
			const auto fi = e2f[i]; 
			if (fi < 0) continue;
			for (int j=0; j<mt.Width(); j++) {
				const auto fj = e2f[j];
				if (fj < 0) continue;

				mt(i,j) += elmat(fi,fj);
			}
		}
	}

	mfem::Array<int> br_vdofs;
	for (int e=0; e<fes.GetNE(); e++) {
		mfem::DenseMatrix ma, d;
		auto &mt = Mt->GetBlock(e);
		Dform.ComputeElementMatrix(e, d);
		Maform.ComputeElementMatrix(e, ma);

		// apply reflection condition 
		rt_br_dofs.GetRow(e, br_vdofs);
		for (int i=0; i<mt.Height(); i++) {
			if (br_tdof_marker[br_vdofs[i]]) {
				for (int j=0; j<mt.Width(); j++) {
					mt(i,j) = 0.0;
					mt(j,i) = 0.0;
				}
				mt(i,i) = 1.0;
			}
		}

		for (int j=0; j<d.Width(); j++) {
			if (br_tdof_marker[br_vdofs[j]]) {
				for (int i=0; i<d.Height(); i++) {
					d(i,j) = 0.0;
				}
			}
		}

		Mt->SetBlock(e, mt);
		D->SetBlock(e, d);
		Ma->SetBlock(e, ma);

		mfem::DenseMatrix g(d, 'T');
		g *= -1.0;
		G->SetBlock(e, g);
	}

	mfem::NormalTraceJumpIntegrator constraint_bfi;
	ifes.ExchangeFaceNbrData();
	auto *Ct = new mfem::SparseMatrix(rt_br_dofs.Size_of_connections(), ifes.GetVSize());
	mfem::Array<int> br_vdofs1, br_vdofs2, i_vdofs;
	for (int f=0; f<mesh.GetNumFaces(); f++) {
		const auto info = mesh.GetFaceInformation(f);
		if (info.IsBoundary()) continue;
		auto *trans = mesh.GetFaceElementTransformations(f);

		rt_br_dofs.GetRow(trans->Elem1No, br_vdofs1);
		const auto length1 = br_vdofs1.Size();
		auto length2 = 0;
		if (!info.IsShared()) {
			rt_br_dofs.GetRow(trans->Elem2No, br_vdofs2);
			length2 = br_vdofs2.Size();
		}

		br_vdofs.SetSize(length1 + length2);
		for (int i=0; i<length1; i++) {
			br_vdofs[i] = br_vdofs1[i];
		}
		if (length2) {
			for (int i=0; i<length2; i++) {
				br_vdofs[i+length1] = br_vdofs2[i];
			}			
		}
		ifes.GetFaceVDofs(f, i_vdofs);

		int elem2no = info.IsShared() ? trans->Elem1No : trans->Elem2No;
		constraint_bfi.AssembleFaceMatrix(*ifes.GetFaceElement(f), 
			*vfes.GetFE(trans->Elem1No), *vfes.GetFE(elem2no), *trans, elmat);
		elmat.Threshold(1e-12 * elmat.MaxMaxNorm());
		Ct->AddSubMatrix(br_vdofs, i_vdofs, elmat, 1);
	}
	Ct->Finalize();

	auto *op = new mfem::BlockOperator(offsets);
	op->SetBlock(0,0, Mt);
	op->SetBlock(0,1, G);
	op->SetBlock(1,0, D);
	op->SetBlock(1,1, Ma);
	op->SetBlock(2,0, new mfem::TransposeOperator(*Ct));
	op->SetBlock(0,2, Ct);
	op->owns_blocks = 1;
	return op;
}

template<typename T>
const T &GetBlock(const mfem::BlockOperator &op, int i, int j)
{
	const T *ptr = dynamic_cast<const T*>(&op.GetBlock(i,j));
	if (!ptr) MFEM_ABORT("operator does not match type");
	return *ptr;
}

void HybridizedRTDiffusionDiscretization::
HatSolver::SetOperator(const mfem::Operator &op)
{
	block_op = dynamic_cast<const mfem::BlockOperator*>(&op);
	if (!block_op) MFEM_ABORT("must be a block operator");

	height = width = block_op->Height();

	const auto &Mt = GetBlock<DenseBlockDiagonalOperator>(*block_op, 0, 0);
	const auto &D = GetBlock<DenseBlockDiagonalOperator>(*block_op, 1, 0);
	const auto &G = GetBlock<DenseBlockDiagonalOperator>(*block_op, 0, 1);
	const auto &Ma = GetBlock<DenseBlockDiagonalOperator>(*block_op, 1, 1);
	const auto &Ct = GetBlock<mfem::SparseMatrix>(*block_op, 0, 2);

	const auto &rt_br_dofs = Mt.GetRowDofs();
	const auto &dg_dofs = Ma.GetRowDofs();

	mfem::DenseMatrix A;
	auto *W = new DenseBlockDiagonalOperator(rt_br_dofs, rt_br_dofs);
	auto *X = new DenseBlockDiagonalOperator(rt_br_dofs, dg_dofs);
	auto *Y = new DenseBlockDiagonalOperator(dg_dofs, rt_br_dofs);
	auto *Z = new DenseBlockDiagonalOperator(dg_dofs, dg_dofs);
	for (int e=0; e<Mt.NumBlocks(); e++) {
		const auto &mt = Mt.GetBlock(e);
		const auto &d = D.GetBlock(e);
		const auto &g = G.GetBlock(e);
		const auto &ma = Ma.GetBlock(e);

		const auto Jsize = mt.Height();
		const auto phi_size = ma.Height();

		A.SetSize(Jsize + phi_size);
		A.SetSubMatrix(0,0, mt);
		A.SetSubMatrix(0,Jsize, g);
		A.SetSubMatrix(Jsize,0, d);
		A.SetSubMatrix(Jsize, Jsize, ma);
		A.Invert();

		auto &w = W->GetBlock(e);
		auto &x = X->GetBlock(e);
		auto &y = Y->GetBlock(e);
		auto &z = Z->GetBlock(e);
		A.GetSubMatrix(0,Jsize, w);
		A.GetSubMatrix(0,Jsize, Jsize, Jsize+phi_size, x);
		A.GetSubMatrix(Jsize,Jsize+phi_size, 0, Jsize, y);
		A.GetSubMatrix(Jsize,Jsize+phi_size, z);
	}

	Ainv = std::make_unique<mfem::BlockOperator>(block_op->RowOffsets());
	Ainv->SetBlock(0,0, W);
	Ainv->SetBlock(0,1, X);
	Ainv->SetBlock(1,0, Y);
	Ainv->SetBlock(1,1, Z);
	Ainv->owns_blocks = 1;

	mfem::SparseMatrix *H_tmp = RAP(*W, Ct);
	auto *Hlocal = new mfem::SparseMatrix(H_tmp->Height());
	{
		auto *I = H_tmp->GetI();
		auto *J = H_tmp->GetJ();
		auto *data = H_tmp->GetData();
		for (int row=0; row<H_tmp->Height(); row++) {
			for (int i=I[row]; i<I[row+1]; i++) {
				const auto col = J[i]; 
				const auto d = data[i];
				Hlocal->Set(row, col, d);
			}
		}
	}
	Hlocal->Finalize(1, true);
	delete H_tmp;

	mfem::HypreParMatrix pH_local(ifes.GetComm(), ifes.GlobalVSize(), ifes.GetDofOffsets(), Hlocal);
	pH_local.SetOwnerFlags(true, true, true);
	H.reset(mfem::RAP(&pH_local, ifes.Dof_TrueDof_Matrix()));

	schur_solver.SetOperator(*H);
}

void HybridizedRTDiffusionDiscretization::
HatSolver::Mult(const mfem::Vector &b, mfem::Vector &x) const
{
	mfem::BlockVector block_x(x, block_op->RowOffsets());
	auto &Jhat = block_x.GetBlock(0);
	auto &phi = block_x.GetBlock(1);
	auto &lam = block_x.GetBlock(2);

	const auto &Ct = GetBlock<mfem::SparseMatrix>(*block_op, 0, 2);

	mfem::BlockVector Ainv_b(Ainv->RowOffsets());
	Ainv_b = 0.0;
	mfem::Vector C_Ainv_b(Ct.Width()), C_Ainv_b_true(ifes.GetTrueVSize()), lam_true(ifes.GetTrueVSize());
	Ainv->Mult(b, Ainv_b);

	Ct.MultTranspose(Ainv_b.GetBlock(0), C_Ainv_b);
	ifes.GetProlongationMatrix()->MultTranspose(C_Ainv_b, C_Ainv_b_true);

	schur_solver.Mult(C_Ainv_b_true, lam_true);
	ifes.GetProlongationMatrix()->Mult(lam_true, lam);

	mfem::Vector Ct_lam(Ct.Height());
	Ct.Mult(lam, Ct_lam);
	Ainv->GetBlock(0,0).Mult(Ct_lam, Jhat);
	add(Ainv_b.GetBlock(0), -1.0, Jhat, Jhat);

	Ainv->GetBlock(1,0).Mult(Ct_lam, phi);
	add(Ainv_b.GetBlock(1), -1.0, phi, phi);
}

void HybridizedRTDiffusionDiscretization::
ProlongationOperator::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
	assert(x.Size() == Width()); 
	assert(y.Size() == Height());
	const mfem::BlockVector block_x(const_cast<mfem::Vector&>(x), col_offsets);
	mfem::BlockVector block_y(y, row_offsets);


	// J -> Jhat 
	mfem::Array<int> vdofs, br_vdofs;
	const auto &Jtrue = block_x.GetBlock(0);
	mfem::Vector J(vfes.GetVSize());
	vfes.GetRestrictionMatrix()->MultTranspose(Jtrue, J);
	auto &Jhat = block_y.GetBlock(0);
	mfem::Array<bool> vdof_marker(vfes.GetVSize());
	vdof_marker = false;
	mfem::Vector elvec;
	for (int e=0; e<vfes.GetNE(); e++) {
		vfes.GetElementVDofs(e, vdofs);
		rt_br_dofs.GetRow(e, br_vdofs);
		J.GetSubVector(vdofs, elvec);
		for (int i=0; i<vdofs.Size(); i++) {
			const auto vdof = mfem::FiniteElementSpace::DecodeDof(vdofs[i]);
			if (vdof_marker[vdof]) { elvec(i) = 0.0; }
			else vdof_marker[vdof] = true;
		}
		Jhat.SetSubVector(br_vdofs, elvec);
	}

	block_y.GetBlock(1) = block_x.GetBlock(1);
	block_y.GetBlock(2) = 0.0;
}

void HybridizedRTDiffusionDiscretization::
ProlongationOperator::MultTranspose(const mfem::Vector &x, mfem::Vector &y) const
{
	const mfem::BlockVector block_x(const_cast<mfem::Vector&>(x), row_offsets);
	mfem::BlockVector block_y(y, col_offsets);

	mfem::Array<int> vdofs, br_vdofs;
	const auto &Jhat = block_x.GetBlock(0);
	mfem::Vector J(vfes.GetVSize());
	for (int e=0; e<vfes.GetNE(); e++) {
		vfes.GetElementVDofs(e, vdofs);
		rt_br_dofs.GetRow(e, br_vdofs);
		for (int i=0; i<vdofs.Size(); i++) {
			const auto vdof = vdofs[i];
			if (vdof >= 0) J(vdof) = Jhat(br_vdofs[i]);
			else J(-1-vdof) = -Jhat(br_vdofs[i]);
		}
	}

	auto &Jtrue = block_y.GetBlock(0);
	vfes.GetRestrictionMatrix()->Mult(J, Jtrue);

	block_y.GetBlock(1) = block_x.GetBlock(1);
}