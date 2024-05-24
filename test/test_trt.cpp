#include "gtest/gtest.h"
#include "lumped_intrule.hpp"
#include "constants.hpp"
#include "block_diag_op.hpp"
#include "transport_op.hpp"
#include "sweep.hpp"

#include "trt_integrators.hpp"
#include "trt_picard.hpp"
#include "trt_linearized.hpp"

TEST(LumpedTRT, Emission1D) {
	auto mesh = mfem::Mesh::MakeCartesian1D(1, 1.0); 
	auto fec = mfem::L2_FECollection(1, mesh.Dimension(), mfem::BasisType::GaussLobatto); 
	auto fes = mfem::FiniteElementSpace(&mesh, &fec); 
	mfem::ConstantCoefficient coef(1.0/constants::StefanBoltzmann); 
	auto nfi = QuadratureLumpedNFIntegrator(new BlackBodyEmissionNFI(coef, 2, 2)); 
	const auto &fe = *fes.GetFE(0); 
	mfem::Vector elfun(fe.GetDof()); 
	elfun(0) = 1.0; 
	elfun(1) = 2.0; 
	mfem::Vector elvec; 
	nfi.AssembleElementVector(fe, *fes.GetElementTransformation(0), elfun, elvec); 
	mfem::Vector ex({0.5, pow(2.0,4)/2}); 
	ex -= elvec; 
	EXPECT_NEAR(ex.Norml2(), 0.0, 1e-14); 
}

TEST(LumpedTRT, Emission2D) {
	auto mesh = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL, false, 1.0, 1.0); 
	auto fec = mfem::L2_FECollection(1, mesh.Dimension(), mfem::BasisType::GaussLobatto); 
	auto fes = mfem::FiniteElementSpace(&mesh, &fec); 
	mfem::ConstantCoefficient coef(1.0/constants::StefanBoltzmann); 
	auto nfi = QuadratureLumpedNFIntegrator(new BlackBodyEmissionNFI(coef, 2, 2)); 
	const auto &fe = *fes.GetFE(0); 
	mfem::Vector elfun(fe.GetDof()); 
	elfun(0) = 0.0; 
	elfun(1) = 1.0; 
	elfun(2) = 2.0; 
	elfun(3) = 3.0; 
	mfem::Vector elvec; 
	nfi.AssembleElementVector(fe, *fes.GetElementTransformation(0), elfun, elvec); 
	mfem::Vector ex({0.0, 0.25, 4.0, pow(3.0, 4)/4}); 
	ex -= elvec; 
	EXPECT_NEAR(ex.Norml2(), 0.0, 1e-13); 
}

TEST(LumpedTRT, EmissionJacobian1D) {
	auto mesh = mfem::Mesh::MakeCartesian1D(1, 1.0); 
	auto fec = mfem::L2_FECollection(1, mesh.Dimension(), mfem::BasisType::GaussLobatto); 
	auto fes = mfem::FiniteElementSpace(&mesh, &fec); 
	mfem::ConstantCoefficient coef(1.0); 
	auto nfi = QuadratureLumpedNFIntegrator(new BlackBodyEmissionNFI(coef, 2, 2)); 
	const auto &fe = *fes.GetFE(0); 
	mfem::Vector elfun(fe.GetDof()); 
	elfun = 1.0; 
	mfem::DenseMatrix grad; 
	nfi.AssembleElementGrad(fe, *fes.GetElementTransformation(0), elfun, grad); 
	grad *= 1.0/constants::StefanBoltzmann; 
	mfem::DenseMatrix ex({{2.0, 0.0}, {0.0, 2.0}}); 
	grad -= ex; 
	double norm = grad.FNorm(); 
	EXPECT_NEAR(norm, 0.0, 1e-14); 
}

TEST(LumpedTRT, EmissionJacobian2D) {
	auto mesh = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL, false, 1.0, 1.0); 
	auto fec = mfem::L2_FECollection(1, mesh.Dimension(), mfem::BasisType::GaussLobatto); 
	auto fes = mfem::FiniteElementSpace(&mesh, &fec); 
	mfem::ConstantCoefficient coef(1.0); 
	auto nfi = QuadratureLumpedNFIntegrator(new BlackBodyEmissionNFI(coef, 2, 2)); 
	const auto &fe = *fes.GetFE(0); 
	mfem::Vector elfun(fe.GetDof()); 
	elfun = 1.0; 
	mfem::DenseMatrix grad; 
	nfi.AssembleElementGrad(fe, *fes.GetElementTransformation(0), elfun, grad); 
	grad *= 1.0/constants::StefanBoltzmann; 
	mfem::DenseMatrix ex(4); 
	ex = 0.0; 
	for (int i=0; i<4; i++) ex(i,i) = 1.0; 
	grad -= ex; 
	double norm = grad.FNorm(); 
	EXPECT_NEAR(norm, 0.0, 1e-14); 
}

TEST(TRT, BlockInverse) {
	auto mesh = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL, false, 1.0, 1.0); 
	auto fec = mfem::L2_FECollection(1, mesh.Dimension(), mfem::BasisType::GaussLobatto); 
	auto fes = mfem::FiniteElementSpace(&mesh, &fec); 
	mfem::ConstantCoefficient coef(1.0); 
	BlockDiagonalByElementNonlinearForm form(&fes); 
	form.AddDomainIntegrator(new BlackBodyEmissionNFI(coef, 2, 2)); 
	form.AddDomainIntegrator(new mfem::MassIntegrator(coef)); 
	mfem::GridFunction temperature(&fes); 
	temperature = 1.0; 
	const auto &grad = form.GetGradient(temperature); 
	mfem::DenseMatrixInverse local_inv; 
	BlockDiagonalByElementSolver inv_grad(local_inv);
	inv_grad.SetOperator(grad);  

	mfem::Vector x(fes.GetVSize()), z(fes.GetVSize()); 
	x.Randomize(); 
	mfem::Vector y(x); 
	grad.Mult(y, z); 
	inv_grad.Mult(z, y); 
	y -= x; 
	EXPECT_NEAR(y.Norml2(), 0.0, 1e-12); 
}

TEST(LumpedTRT, BlockInverse) {
	auto mesh = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL, false, 1.0, 1.0); 
	auto fec = mfem::L2_FECollection(1, mesh.Dimension(), mfem::BasisType::GaussLobatto); 
	auto fes = mfem::FiniteElementSpace(&mesh, &fec); 
	mfem::ConstantCoefficient coef(1.0); 
	BlockDiagonalByElementNonlinearForm form(&fes); 
	form.AddDomainIntegrator(new QuadratureLumpedNFIntegrator(new BlackBodyEmissionNFI(coef))); 
	form.AddDomainIntegrator(new QuadratureLumpedNFIntegrator(new mfem::MassIntegrator(coef))); 
	mfem::GridFunction temperature(&fes); 
	temperature = 1.0; 
	const auto &grad = form.GetGradient(temperature); 
	DiagonalDenseMatrixInverse local_inv; 
	BlockDiagonalByElementSolver inv_grad(local_inv); 
	inv_grad.SetOperator(grad); 
	mfem::Vector x(fes.GetVSize()), z(fes.GetVSize()); 
	x.Randomize(); 
	mfem::Vector y(x); 
	grad.Mult(y, z); 
	inv_grad.Mult(z, y); 
	y -= x; 
	EXPECT_NEAR(y.Norml2(), 0.0, 1e-12); 
}

// solve T^4 - T = 0 <=> T (T^3 - 1) = 0 => solutions are 0 and +1 
TEST(LumpedTRT, NewtonSolve) {
	auto mesh = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL, false, 1.0, 1.0); 
	auto fec = mfem::L2_FECollection(1, mesh.Dimension(), mfem::BasisType::GaussLobatto); 
	auto fes = mfem::FiniteElementSpace(&mesh, &fec); 
	mfem::ConstantCoefficient sbi(1.0/constants::StefanBoltzmann); 
	mfem::ConstantCoefficient none(-1.0); 
	BlockDiagonalByElementNonlinearForm form(&fes); 
	form.AddDomainIntegrator(new QuadratureLumpedNFIntegrator(new BlackBodyEmissionNFI(sbi))); 
	form.AddDomainIntegrator(new QuadratureLumpedNFIntegrator(new mfem::MassIntegrator(none))); 

	mfem::GridFunction T(&fes); 
	T(0) = 12.0; T(1) = 7.3; T(2) = 9.6; T(3) = 10.0; 

	mfem::NewtonSolver newton; 
	DiagonalDenseMatrixInverse local_inv; 
	newton.SetSolver(local_inv); 
	newton.SetAbsTol(1e-12); 
	newton.SetMaxIter(20); 
	BlockDiagonalByElementNonlinearSolver solver(newton); 
	solver.SetOperator(form); 
	mfem::Vector blank; 
	solver.Mult(blank, T); 
	EXPECT_TRUE(solver.GetConverged()); 

	mfem::Vector ones(fes.GetVSize()); 
	ones = 1.0; 
	T -= ones; 
	EXPECT_NEAR(T.Norml2(), 0.0, 1e-12); 
}

TEST(TRT, NewtonSolve) {
	auto mesh = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL, false, 1.0, 1.0); 
	auto fec = mfem::L2_FECollection(1, mesh.Dimension(), mfem::BasisType::GaussLobatto); 
	auto fes = mfem::FiniteElementSpace(&mesh, &fec); 
	mfem::ConstantCoefficient sbi(1.0/constants::StefanBoltzmann); 
	mfem::ConstantCoefficient none(-1.0); 
	BlockDiagonalByElementNonlinearForm form(&fes); 
	form.AddDomainIntegrator(new BlackBodyEmissionNFI(sbi, 2, 2)); 
	form.AddDomainIntegrator(new mfem::MassIntegrator(none)); 

	mfem::GridFunction T(&fes); 
	T(0) = 12.0; T(1) = 7.3; T(2) = 9.6; T(3) = 10.0; 

	mfem::NewtonSolver newton; 
	mfem::DenseMatrixInverse local_inv; 
	newton.SetSolver(local_inv); 
	newton.SetAbsTol(1e-12); 
	newton.SetMaxIter(20); 
	BlockDiagonalByElementNonlinearSolver solver(newton); 
	solver.SetOperator(form); 
	mfem::Vector blank;
	solver.Mult(blank, T); 
	EXPECT_TRUE(solver.GetConverged()); 

	mfem::Vector ones(fes.GetVSize()); 
	ones = 1.0; 
	T -= ones; 
	EXPECT_NEAR(T.Norml2(), 0.0, 1e-12); 
}

#ifdef MFEM_USE_SUNDIALS 
TEST(TRT, SundialsSolve) {
	auto mesh = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL, false, 1.0, 1.0); 
	auto fec = mfem::L2_FECollection(1, mesh.Dimension(), mfem::BasisType::GaussLobatto); 
	auto fes = mfem::FiniteElementSpace(&mesh, &fec); 
	mfem::ConstantCoefficient sbi(1.0/constants::StefanBoltzmann); 
	mfem::ConstantCoefficient none(-1.0); 
	BlockDiagonalByElementNonlinearForm form(&fes); 
	form.AddDomainIntegrator(new BlackBodyEmissionNFI(sbi, 2, 2)); 
	form.AddDomainIntegrator(new mfem::MassIntegrator(none)); 

	mfem::GridFunction T(&fes); 
	T(0) = 12.0; T(1) = 7.3; T(2) = 9.6; T(3) = 10.0; 

	mfem::KINSolver solver(KIN_NONE); 
	solver.SetOperator(form); 
	mfem::DenseMatrixInverse local_inv; 
	BlockDiagonalByElementSolver grad_inv(local_inv); 
	solver.SetSolver(grad_inv); 
	solver.SetAbsTol(1e-12); 
	solver.SetRelTol(1e-12); 
	solver.SetMaxIter(40); 
	solver.SetPrintLevel(0); 
	solver.SetMaxSetupCalls(1); 
	solver.iterative_mode = true;
	mfem::Vector blank; 
	solver.Mult(blank, T); 

	mfem::Vector ones(fes.GetVSize()); 
	ones = 1.0; 
	T -= ones; 
	EXPECT_NEAR(T.Norml2(), 0.0, 1e-12); 	
}
#endif

TEST(TRT, BlockDiagResidual) {
	auto mesh = mfem::Mesh::MakeCartesian2D(3, 3, mfem::Element::QUADRILATERAL, false, 1.0, 1.0); 
	auto fec = mfem::L2_FECollection(1, mesh.Dimension(), mfem::BasisType::GaussLobatto); 
	auto fes = mfem::FiniteElementSpace(&mesh, &fec); 
	mfem::ConstantCoefficient sbi(1.0/constants::StefanBoltzmann); 
	mfem::ConstantCoefficient none(-1.0); 
	BlockDiagonalByElementNonlinearForm form(&fes); 
	form.AddDomainIntegrator(new BlackBodyEmissionNFI(sbi, 2, 2)); 
	form.AddDomainIntegrator(new mfem::MassIntegrator(none)); 

	mfem::GridFunction T(&fes); 
	T.Randomize(); 
	mfem::Vector f1(T.Size()); 
	form.Mult(T, f1); 

	mfem::Array<int> vdofs; 
	mfem::Vector subx, subf; 
	mfem::Vector f2(T.Size()); 
	for (int e=0; e<fes.GetNE(); e++) {
		fes.GetElementVDofs(e, vdofs); 
		T.GetSubVector(vdofs, subx); 
		subf.SetSize(vdofs.Size()); 
		form.AssembleLocalResidual(e, subx, subf); 
		f2.SetSubVector(vdofs, subf); 
	}
	f1 -= f2; 
	EXPECT_NEAR(f1.Norml2(), 0.0, 1e-12); 
}

TEST(TRT, BlockDiagGradient) {
	auto mesh = mfem::Mesh::MakeCartesian2D(3, 3, mfem::Element::QUADRILATERAL, false, 1.0, 1.0); 
	auto fec = mfem::L2_FECollection(1, mesh.Dimension(), mfem::BasisType::GaussLobatto); 
	auto fes = mfem::FiniteElementSpace(&mesh, &fec); 
	mfem::ConstantCoefficient sbi(1.0/constants::StefanBoltzmann); 
	mfem::ConstantCoefficient none(-1.0); 
	BlockDiagonalByElementNonlinearForm form(&fes); 
	form.AddDomainIntegrator(new BlackBodyEmissionNFI(sbi, 2, 2)); 
	form.AddDomainIntegrator(new mfem::MassIntegrator(none)); 

	mfem::GridFunction T(&fes); 
	T.Randomize(); 

	const auto *grad = dynamic_cast<BlockDiagonalByElementOperator*>(&form.GetGradient(T)); 

	mfem::Array<int> vdofs; 
	mfem::Vector subx;
	mfem::DenseMatrix elmat; 
	double sum = 0.0; 
	for (int e=0; e<fes.GetNE(); e++) {
		fes.GetElementVDofs(e, vdofs); 
		T.GetSubVector(vdofs, subx); 
		form.AssembleLocalGradient(e, subx, elmat); 
		const auto &elmat2 = grad->GetElementMatrix(e); 
		elmat -= elmat2; 
		sum += elmat.FNorm(); 
	}
	EXPECT_NEAR(sum, 0.0, 1e-12); 
}

TEST(TRT, LocalTemperatureSolve) {
	auto smesh = mfem::Mesh::MakeCartesian1D(10, 1.0); 
	auto mesh = mfem::ParMesh(MPI_COMM_WORLD, smesh); 
	auto fec = mfem::L2_FECollection(1, mesh.Dimension(), mfem::BasisType::GaussLobatto); 
	auto fes = mfem::ParFiniteElementSpace(&mesh, &fec); 
	mfem::ConstantCoefficient sigma(1e3); 
	mfem::ConstantCoefficient Cvdt(1e10/1e-9); 
	BlockDiagonalByElementNonlinearForm form(&fes); 
	form.AddDomainIntegrator(new BlackBodyEmissionNFI(sigma, 2, 2)); 
	form.AddDomainIntegrator(new mfem::MassIntegrator(Cvdt)); 

	mfem::BilinearForm Msigma(&fes); 
	Msigma.AddDomainIntegrator(new mfem::MassIntegrator(sigma)); 
	Msigma.Assemble(); 
	Msigma.Finalize(); 

	mfem::BilinearForm M(&fes); 
	M.AddDomainIntegrator(new mfem::MassIntegrator); 
	M.Assemble(); 
	M.Finalize(); 

	mfem::ParGridFunction phi(&fes), T(&fes); 
	phi = constants::StefanBoltzmann * 1e4; // 10 eV radiation temperature 
	T = 0.025; // room temp 
	mfem::Vector source(fes.GetVSize()); 
	Msigma.Mult(phi, source); 
	M.AddMult(T, source); 

	mfem::DenseMatrixInverse local_mat_inv; 
	EnergyBalanceNewtonSolver local_solver; 
	local_solver.SetSolver(local_mat_inv); 
	local_solver.SetRelTol(1e-6); 
	local_solver.SetMaxIter(40); 
	local_solver.SetPrintLevel(-1); 
	BlockDiagonalByElementNonlinearSolver solver(local_solver); 
	solver.SetOperator(form); 
	mfem::Vector f(fes.GetVSize()); 
	form.Mult(T, f); 
	f -= source; 
	double initial_norm = sqrt(mfem::InnerProduct(MPI_COMM_WORLD, f, f));
	solver.Mult(source, T); 
	EXPECT_TRUE(solver.GetConverged()); 

	form.Mult(T, f);
	f -= source; 
	double norm = sqrt(mfem::InnerProduct(MPI_COMM_WORLD, f, f));
	EXPECT_LT(solver.GetFinalRelNorm(), 1e-6); 
}

TEST(NewtonTRT, JacobianSolver) {
	const int fe_order = 1; 
	auto smesh = mfem::Mesh::MakeCartesian1D(3, 1.0); 
	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh); 
	const auto dim = mesh.Dimension();
	LevelSymmetricQuadrature quad(6, dim); 

	mfem::L2_FECollection fec(fe_order, dim, mfem::BasisType::GaussLobatto); 
	mfem::ParFiniteElementSpace fes(&mesh, &fec); 

	double total_val = 1.0; 
	double scattering_val = .25; 
	mfem::ConstantCoefficient total(total_val);
	mfem::ConstantCoefficient scattering(scattering_val);  
	mfem::ConstantCoefficient inflow(2.0/4/M_PI); 

	TransportVectorExtents psi_ext(1, quad.Size(), fes.GetVSize()); 
	MomentVectorExtents phi_ext(1, 1, fes.GetVSize()); 
	const auto psi_size = TotalExtent(psi_ext); 
	mfem::Vector psi(psi_size); 
	psi = 1.0/4/M_PI; 

	DiscreteToMoment D(quad, psi_ext, phi_ext); 

	mfem::ConstantCoefficient Cvdt(1.0); 
	BlockDiagonalByElementNonlinearForm meb_form(&fes);
	meb_form.AddDomainIntegrator(new BlackBodyEmissionNFI(total, 2, 2)); 
	meb_form.AddDomainIntegrator(new mfem::MassIntegrator(Cvdt)); 

	BlockDiagonalByElementNonlinearForm emission_form(&fes); 
	emission_form.AddDomainIntegrator(new BlackBodyEmissionNFI(total, 2, 2));

	mfem::BilinearForm Mtot(&fes); 
	Mtot.AddDomainIntegrator(new mfem::MassIntegrator(total)); 
	Mtot.Assemble(); 
	Mtot.Finalize(); 

	mfem::GridFunction total_data(&fes); 
	total_data.ProjectCoefficient(total); 
	InverseAdvectionOperator Linv(fes, quad, total_data); 

	mfem::Array<int> offsets(3); 
	offsets[0] = 0; 
	offsets[1] = fes.GetVSize(); 
	offsets[2] = fes.GetVSize(); 
	offsets.PartialSum(); 

	mfem::DenseMatrixInverse local_inv; 
	BlockDiagonalByElementSolver meb_grad_inv(local_inv); 

	mfem::GMRESSolver gmres(MPI_COMM_WORLD); 
	gmres.SetAbsTol(1e-6); 
	gmres.SetMaxIter(100); 
	gmres.SetPrintLevel(0); 

	NewtonTRTOperator::NonlinearOperator op(offsets, Linv, D, emission_form, meb_form, Mtot, psi); 
	NewtonTRTOperator::JacobianSolver grad_inv(offsets, gmres, meb_grad_inv); 

	mfem::BlockVector x(offsets), y(offsets), z(offsets), source(offsets); 
	mfem::ParGridFunction phi(&fes, x.GetBlock(0), 0); 
	mfem::ParGridFunction T(&fes, x.GetBlock(1), 0); 
	T = 0.025; 
	phi = constants::StefanBoltzmann * pow(0.025, 4); 
	source.Randomize(12345); 
	auto &grad = op.GetGradient(x); 
	grad_inv.SetOperator(grad); 
	grad.Mult(source, y); 
	grad_inv.Mult(y, z); 
	z -= source; 
	EXPECT_NEAR(z.Norml2(), 0.0, 1e-5); 
}