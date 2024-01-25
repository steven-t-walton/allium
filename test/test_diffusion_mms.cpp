#include "gtest/gtest.h"
#include "p1diffusion.hpp"

#ifdef MFEM_USE_SUITESPARSE
// use exact solution 
double P1DiffusionError1D(int Ne, int fe_order) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D(Ne, 1.0); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(fe_order, dim, mfem::BasisType::GaussLegendre); 
	mfem::FiniteElementSpace fes(&mesh, &fec); 
	mfem::FiniteElementSpace vfes(&mesh, &fec, dim); 

	mfem::Array<int> offsets(3); 
	offsets[0] = 0; 
	offsets[1] = vfes.GetVSize(); 
	offsets[2] = fes.GetVSize(); 
	offsets.PartialSum(); 

	mfem::ConstantCoefficient total(1.0); 
	mfem::ConstantCoefficient absorption(0.0); 

	mfem::BilinearForm Mtform(&vfes);
	mfem::ProductCoefficient total3(3.0, total); 
	Mtform.AddDomainIntegrator(new mfem::VectorMassIntegrator(total3));
	Mtform.AddInteriorFaceIntegrator(new DGVectorJumpJumpIntegrator); 
	Mtform.AddBdrFaceIntegrator(new DGVectorJumpJumpIntegrator); 
	Mtform.Assemble(); 
	Mtform.Finalize();  

	mfem::BilinearForm Maform(&fes); 
	mfem::ConstantCoefficient alpha(0.25); 
	Maform.AddDomainIntegrator(new mfem::MassIntegrator(absorption)); 
	Maform.AddInteriorFaceIntegrator(new PenaltyIntegrator(4, false)); 
	Maform.AddBdrFaceIntegrator(new mfem::BoundaryMassIntegrator(alpha)); 
	Maform.Assemble(); 
	Maform.Finalize(); 

	mfem::MixedBilinearForm Dform(&vfes, &fes); 
	mfem::ConstantCoefficient neg_one(-1.0); 
	Dform.AddDomainIntegrator(new mfem::TransposeIntegrator(new mfem::GradientIntegrator(neg_one))); 
	Dform.AddInteriorFaceIntegrator(new DGJumpAverageIntegrator); 
	Dform.AddBdrFaceIntegrator(new DGJumpAverageIntegrator(0.5)); 
	Dform.Assemble(); 
	Dform.Finalize(); 
	auto DT = std::unique_ptr<mfem::SparseMatrix>(Transpose(Dform.SpMat())); 
	(*DT) *= -1.0; 

	mfem::LinearForm fform(&fes); 
	auto source = [](const mfem::Vector &x) {
		return 1.0; 
	};
	mfem::FunctionCoefficient source_coef(source); 
	fform.AddDomainIntegrator(new mfem::DomainLFIntegrator(source_coef)); 
	fform.Assemble(); 


	mfem::BlockVector b(offsets), x(offsets); 
	b.GetBlock(0) = 0.0; 
	b.GetBlock(1) = fform; 

	mfem::BlockMatrix A(offsets); 
	A.SetBlock(0,0, &Mtform.SpMat()); 
	A.SetBlock(1,0, &Dform.SpMat()); 
	A.SetBlock(0,1, DT.get()); 
	A.SetBlock(1,1, &Maform.SpMat()); 
	A.Finalize(); 
	auto mono = std::unique_ptr<mfem::SparseMatrix>(A.CreateMonolithic()); 

	mfem::KLUSolver solver(*mono); 
	solver.Mult(b, x); 

	mfem::GridFunction u(&fes, x.GetBlock(1)); 
	mfem::GridFunction v(&vfes, x.GetBlock(0)); 

	auto exact_func = [](const mfem::Vector &x) {
		return 1 + 3*x(0)/2 - 3*x(0)*x(0)/2; 
	}; 
	mfem::FunctionCoefficient exact_coef(exact_func); 
	double err = u.ComputeL2Error(exact_coef); 

	return err; 
}

TEST(MMS, P1Diffusion1Dp1) {
	const auto fe_order = 1; 
	double E1 = P1DiffusionError1D(10, fe_order); 
	double E2 = P1DiffusionError1D(20, fe_order); 
	double ooa = log2(E1/E2); 
	EXPECT_TRUE(std::fabs(fe_order+1-ooa)<.2); 
}

// exact solution is quadratic => error should be zero independent of h 
TEST(MMS, P1Diffusion1Dp2) {
	const auto fe_order = 2; 
	double E = P1DiffusionError1D(10, fe_order); 
	EXPECT_TRUE(E < 1e-12); 
}
#endif

using HypreParMatrixPtr = std::unique_ptr<mfem::HypreParMatrix>; 
double LDGError(mfem::Mesh &smesh, int fe_order, 
	std::function<double(const mfem::Vector&)> qmms, std::function<double(const mfem::Vector&)> exsol) 
{
	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(fe_order, dim, mfem::BasisType::GaussLegendre); 
	mfem::ParFiniteElementSpace fes(&mesh, &fec); 
	mfem::ParFiniteElementSpace vfes(&mesh, &fec, dim); // dim copies of scalar space 

	mfem::Array<int> offsets(3); 
	offsets[0] = 0; 
	offsets[1] = vfes.GetVSize(); 
	offsets[2] = fes.GetVSize(); 
	offsets.PartialSum(); 

	mfem::ConstantCoefficient total(1.0); 
	mfem::ProductCoefficient total3(3.0, total); 
	mfem::ParBilinearForm iMtform(&vfes); 
	iMtform.AddDomainIntegrator(new mfem::InverseIntegrator(new mfem::VectorMassIntegrator(total3))); 
	iMtform.Assemble(); 
	iMtform.Finalize(); 
	auto iMt = HypreParMatrixPtr(iMtform.ParallelAssemble()); 

	mfem::ParBilinearForm Maform(&fes); 
	mfem::ConstantCoefficient alpha(0.25); 
	Maform.AddInteriorFaceIntegrator(new PenaltyIntegrator(.25, false)); 
	Maform.AddBdrFaceIntegrator(new PenaltyIntegrator(pow(fe_order+1,2), true)); // scale on bdr to get Dirichlet bc 
	Maform.Assemble(); 
	Maform.Finalize(); 
	auto Ma = HypreParMatrixPtr(Maform.ParallelAssemble()); 

	mfem::ParMixedBilinearForm Dform(&vfes, &fes); 
	mfem::ConstantCoefficient neg_one(-1.0); 
	Dform.AddDomainIntegrator(new mfem::TransposeIntegrator(new mfem::GradientIntegrator(neg_one))); 
	mfem::Vector beta(dim); 
	// arbitrary direction not aligned with axis 
	for (auto d=0; d<dim; d++) {
		beta(d) = d+1; 
	}
	Dform.AddInteriorFaceIntegrator(new mfem::LDGTraceIntegrator(&beta)); 
	Dform.AddBdrFaceIntegrator(new mfem::LDGTraceIntegrator); 
	Dform.Assemble(); 
	Dform.Finalize(); 
	auto D = HypreParMatrixPtr(Dform.ParallelAssemble()); 
	auto DT = HypreParMatrixPtr(D->Transpose()); 

	auto iMtDT = HypreParMatrixPtr(mfem::ParMult(iMt.get(), DT.get(), true)); 
	auto DiMtDT = HypreParMatrixPtr(mfem::ParMult(D.get(), iMtDT.get(), true)); 
	auto S = HypreParMatrixPtr(mfem::ParAdd(DiMtDT.get(), Ma.get())); 

	mfem::ParLinearForm fform(&fes); 
	mfem::FunctionCoefficient source_coef(qmms); 
	fform.AddDomainIntegrator(new mfem::DomainLFIntegrator(source_coef)); 
	fform.Assemble(); 

	mfem::BlockVector b(offsets), x(offsets); 
	b.GetBlock(0) = 0.0; 
	b.GetBlock(1) = fform; 
	x = 0.0; 

	mfem::BiCGSTABSolver solver(MPI_COMM_WORLD); 
	solver.SetAbsTol(1e-10); 
	solver.SetMaxIter(100); 
	solver.SetPrintLevel(0); 
	solver.SetOperator(*S); 
	mfem::HypreBoomerAMG amg(*S); 
	amg.SetPrintLevel(0); 
	solver.SetPreconditioner(amg); 
	solver.Mult(b.GetBlock(1), x.GetBlock(1)); 
	EXPECT_TRUE(solver.GetConverged()); 

	mfem::ParGridFunction u(&fes, x.GetBlock(1).GetData()); 
	mfem::FunctionCoefficient exsol_coef(exsol); 
	auto err = u.ComputeL2Error(exsol_coef); 
	return err; 
} 

auto qmms2d = [](const mfem::Vector &x) {
	return 2./3*M_PI*M_PI*sin(M_PI*x(0))*sin(M_PI*x(1)); 
}; 
auto exsol2d = [](const mfem::Vector &x) {
	return sin(M_PI*x(0))*sin(M_PI*x(1)); 
};

TEST(MMS, LDGDiffusion2Dp1) {
	auto fe_order = 1; 
	auto Ne = 10; 
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(Ne, Ne, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 	
	double E1 = LDGError(mesh, fe_order, qmms2d, exsol2d); 
	mesh.UniformRefinement(); 
	double E2 = LDGError(mesh, fe_order, qmms2d, exsol2d); 
	double ooa = log2(E1/E2); 
	EXPECT_TRUE(std::fabs(fe_order+1 - ooa)<.1); 
}

TEST(MMS, LDGDiffusion2Dp2) {
	auto fe_order = 2; 
	auto Ne = 5; 
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(Ne, Ne, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 	
	double E1 = LDGError(mesh, fe_order, qmms2d, exsol2d); 
	mesh.UniformRefinement(); 
	double E2 = LDGError(mesh, fe_order, qmms2d, exsol2d); 
	double ooa = log2(E1/E2); 
	EXPECT_TRUE(std::fabs(fe_order+1 - ooa)<.1); 
}

TEST(MMS, LDGDiffusion2Dp3) {
	auto fe_order = 3; 
	auto Ne = 3; 
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(Ne, Ne, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 	
	double E1 = LDGError(mesh, fe_order, qmms2d, exsol2d); 
	mesh.UniformRefinement(); 
	double E2 = LDGError(mesh, fe_order, qmms2d, exsol2d); 
	double ooa = log2(E1/E2); 
	EXPECT_TRUE(std::fabs(fe_order+1 - ooa)<.1); 
}

auto qmms3d = [](const mfem::Vector &x) {
	return M_PI*M_PI*sin(M_PI*x(0))*sin(M_PI*x(1))*sin(M_PI*x(2)); 
}; 
auto exsol3d = [](const mfem::Vector &x) {
	return sin(M_PI*x(0))*sin(M_PI*x(1))*sin(M_PI*x(2)); 
};

TEST(MMS, LDGDiffusion3Dp1) {
	auto fe_order = 1; 
	auto Ne = 5; 
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(Ne, Ne, Ne, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0, false); 	
	double E1 = LDGError(mesh, fe_order, qmms3d, exsol3d); 
	mesh.UniformRefinement(); 
	double E2 = LDGError(mesh, fe_order, qmms3d, exsol3d); 
	double ooa = log2(E1/E2); 
	EXPECT_TRUE(std::fabs(fe_order+1 - ooa)<.1); 
}

