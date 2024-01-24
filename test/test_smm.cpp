#include "gtest/gtest.h"
#include "p1diffusion.hpp"
#include "smm_integrators.hpp"
#include "phase_coefficient.hpp"
#include "linalg.hpp"

TEST(SMM, CorrectionTensorIsotropic) {
	mfem::Mesh smesh = mfem::Mesh::MakeCartesian2D(5,5,mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(1, dim, mfem::BasisType::GaussLegendre); 
	mfem::ParFiniteElementSpace fes(&mesh, &fec); 
	LevelSymmetricQuadrature quad(4, dim); 

	TransportVectorExtents psi_ext(1,quad.Size(),fes.GetVSize()); 
	mfem::Vector psi(TotalExtent(psi_ext)); 
	auto f = [](const mfem::Vector &x, const mfem::Vector &Omega) {
		return sin(M_PI*x(0))*sin(M_PI*x(1)); 
	};
	FunctionGrayCoefficient psi_coef(f); 
	TransportVectorView psi_view(psi.GetData(), psi_ext); 
	ProjectPsi(fes, quad, psi_coef, psi_view); 
	SMMCorrectionTensorCoefficient T(fes, quad, psi_view);
	mfem::DenseMatrix t(dim); 
	double norm = 0.0; 
	for (auto e=0; e<mesh.GetNE(); e++) {
		auto &trans = *mesh.GetElementTransformation(e); 
		int geom = mesh.GetElementBaseGeometry(e);		
		auto &ref_cent = mfem::Geometries.GetCenter(geom);
		T.Eval(t, trans, ref_cent); 
		norm += t.FNorm(); 
	}

	EXPECT_NEAR(norm, 0.0, 1e-14); 
}

TEST(SMM, CorrectionTensorQuadratic) {
	mfem::Mesh smesh = mfem::Mesh::MakeCartesian2D(5,5,mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(1, dim, mfem::BasisType::GaussLegendre); 
	mfem::ParFiniteElementSpace fes(&mesh, &fec); 
	LevelSymmetricQuadrature quad(4, dim); 

	TransportVectorExtents psi_ext(1,quad.Size(),fes.GetVSize()); 
	mfem::Vector psi(TotalExtent(psi_ext)); 
	auto f = [](const mfem::Vector &x, const mfem::Vector &Omega) {
		return Omega(0)*Omega(0) + Omega(1)*Omega(1) + Omega(0)*Omega(1); 
	};
	FunctionGrayCoefficient psi_coef(f); 
	TransportVectorView psi_view(psi.GetData(), psi_ext); 
	ProjectPsi(fes, quad, psi_coef, psi_view); 

	SMMCorrectionTensorCoefficient T(fes, quad, psi_view);
	mfem::DenseMatrix t(dim); 
	T.Eval(t, *mesh.GetElementTransformation(0), mfem::Geometries.GetCenter(mesh.GetElementBaseGeometry(0))); 
	mfem::DenseMatrix ex({{8*M_PI/45, 4*M_PI/15}, {4*M_PI/15, 8*M_PI/45}}); 
	ex -= t; 
	double norm = ex.FNorm(); 
	EXPECT_NEAR(norm, 0.0, 1e-13); 
}

TEST(SMM, BdrCorrectionLinear) {
	mfem::Mesh smesh = mfem::Mesh::MakeCartesian2D(2,2, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh); 
	const auto dim = mesh.Dimension(); 
	
	mfem::L2_FECollection fec(1, dim, mfem::BasisType::GaussLegendre); 
	mfem::ParFiniteElementSpace fes(&mesh, &fec); 
	LevelSymmetricQuadrature quad(4, dim); 

	TransportVectorExtents psi_ext(1,quad.Size(),fes.GetVSize()); 
	mfem::Vector psi(TotalExtent(psi_ext)); 
	auto f = [](const mfem::Vector &x, const mfem::Vector &Omega) {
		return 5.0 + Omega(0) + Omega(1); 
	};
	FunctionGrayCoefficient psi_coef(f); 
	TransportVectorView psi_view(psi.GetData(), psi_ext); 
	ProjectPsi(fes, quad, psi_coef, psi_view); 

	mfem::Vector nor(dim); 
	nor = 0.0; 
	nor(0) = 1.0; 
	auto alpha = ComputeAlpha(quad, nor); 
	SMMBdrCorrectionFactorCoefficient beta_coef(fes, quad, psi_view, alpha); 
	auto &trans = *mesh.GetFaceElementTransformations(0); 
	const auto &ref_cent = mfem::Geometries.GetCenter(trans.GetGeometryType()); 
	auto beta = beta_coef.Eval(trans, ref_cent); 
	EXPECT_NEAR(beta, 0.0, 1e-14); 
}

// absolute value in beta => can't integrate exactly 
// verify that it converges roughly linearly in #angles 
std::tuple<double,int> BetaError(int sn_order) {
	mfem::Mesh smesh = mfem::Mesh::MakeCartesian2D(2,2, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh); 
	const auto dim = mesh.Dimension(); 
	
	mfem::L2_FECollection fec(1, dim, mfem::BasisType::GaussLegendre); 
	mfem::ParFiniteElementSpace fes(&mesh, &fec); 
	LevelSymmetricQuadrature quad(sn_order, dim); 

	TransportVectorExtents psi_ext(1,quad.Size(),fes.GetVSize()); 
	mfem::Vector psi(TotalExtent(psi_ext)); 
	auto f = [](const mfem::Vector &x, const mfem::Vector &Omega) {
		return Omega(0)*Omega(0) + Omega(1)*Omega(1) + Omega(0)*Omega(1); 
	};
	FunctionGrayCoefficient psi_coef(f); 
	TransportVectorView psi_view(psi.GetData(), psi_ext); 
	ProjectPsi(fes, quad, psi_coef, psi_view); 

	mfem::Vector nor(dim); 
	nor = 0.0; 
	nor(0) = 1.0; 
	auto alpha = ComputeAlpha(quad, nor); 
	SMMBdrCorrectionFactorCoefficient beta_coef(fes, quad, psi_view, alpha); 
	auto &trans = *mesh.GetFaceElementTransformations(0); 
	const auto &ref_cent = mfem::Geometries.GetCenter(trans.GetGeometryType()); 
	auto beta = beta_coef.Eval(trans, ref_cent); 
	auto err = std::fabs(beta - M_PI/6); 
	return {err, quad.Size()}; 
}

TEST(SMM, BdrCorrectionQuadratic) {
	auto [E1,N1] = BetaError(12); 
	auto [E2,N2] = BetaError(24); 
	double ooa = log(E1/E2) / log((double)N2/N1); 
	EXPECT_NEAR(ooa, 1.0, .3); 
}

std::tuple<double,double> IndependentLDGSMMError(int Ne, int fe_order) {
	double delta = 0.5; 
	double gamma = 1.0; 
	double base = 2.0; 
	auto sn_order = 4; 
	mfem::Mesh smesh = mfem::Mesh::MakeCartesian2D(Ne, Ne, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh); 
	const auto dim = mesh.Dimension(); 

	double total = 1.0; 
	double scattering = .25; 
	
	mfem::L2_FECollection fec(fe_order, dim, mfem::BasisType::GaussLegendre); 
	mfem::ParFiniteElementSpace fes(&mesh, &fec); 
	mfem::ParFiniteElementSpace vfes(&mesh, &fec, dim); 
	LevelSymmetricQuadrature quad(sn_order, dim); 

	TransportVectorExtents psi_ext(1,quad.Size(),fes.GetVSize()); 
	mfem::Vector psi(TotalExtent(psi_ext)); 
	auto f = [&delta, &gamma, base](const mfem::Vector &x, const mfem::Vector &Omega) {
		return (sin(M_PI*x(0))*sin(M_PI*x(1)) + delta*(Omega(0) + Omega(1))*sin(2*M_PI*x(0))*sin(2*M_PI*x(1)) 
			+ gamma*(Omega*Omega)*sin(3*M_PI*x(0))*sin(3*M_PI*x(1)) + base)/4/M_PI; 
	};
	FunctionGrayCoefficient psi_coef(f); 
	TransportVectorView psi_view(psi.GetData(), psi_ext); 
	ProjectPsi(fes, quad, psi_coef, psi_view); 

	mfem::Vector nor(dim); 
	nor = 0.0; 
	nor(0) = 1.0; 
	const auto alpha = ComputeAlpha(quad, nor); 
	SMMCorrectionTensorCoefficient T(fes, quad, psi_view); 
	SMMBdrCorrectionFactorCoefficient beta(fes, quad, psi_view, alpha); 	

	auto Q0f = [&total, &scattering, &delta, &gamma, base](const mfem::Vector &x) {
		return (total-scattering)*(sin(M_PI*x(0))*sin(M_PI*x(1)) + base + gamma*2./3*sin(3*M_PI*x(0))*sin(3*M_PI*x(1))) 
			+ delta*2*M_PI/3*sin(2*M_PI*(x(0) + x(1))); 
	};
	mfem::FunctionCoefficient Q0(Q0f); 
	auto Q1f = [&total, &delta, &gamma](const mfem::Vector &x, mfem::Vector &v) {
		v.SetSize(x.Size()); 
		v(0) = M_PI*(cos(M_PI*x(0))*sin(M_PI*x(1))/3 + gamma*4./5*cos(3*M_PI*x(0))*sin(3*M_PI*x(1))); 
		v(1) = M_PI*(sin(M_PI*x(0))*cos(M_PI*x(1))/3 + gamma*4./5*sin(3*M_PI*x(0))*cos(3*M_PI*x(1))); 
		v += delta*total/3*sin(2*M_PI*x(0))*sin(2*M_PI*x(1)); 
	};
	mfem::VectorFunctionCoefficient Q1(dim, Q1f); 

	mfem::ParLinearForm fform(&fes); 
	fform.AddDomainIntegrator(new mfem::DomainLFIntegrator(Q0)); 
	InflowPartialCurrentCoefficient Jin(psi_coef, quad); 
	mfem::SumCoefficient bdr_coef_f(beta, Jin, -0.5, -1.0); 
	fform.AddBdrFaceIntegrator(new mfem::BoundaryLFIntegrator(bdr_coef_f, 2, 1)); 
	fform.Assemble(); 

	mfem::ParLinearForm gform(&vfes); 
	gform.AddDomainIntegrator(new mfem::VectorDomainLFIntegrator(Q1)); 
	gform.AddDomainIntegrator(new WeakTensorDivergenceLFIntegrator(T)); 
	gform.AddInteriorFaceIntegrator(new VectorJumpTensorAverageLFIntegrator(T)); 
	gform.AddBdrFaceIntegrator(new VectorJumpTensorAverageLFIntegrator(T)); 
	mfem::SumCoefficient bdr_coef_g(beta, Jin, 1./2/alpha/3, 1./alpha/3);
	gform.AddBdrFaceIntegrator(new BoundaryNormalFaceLFIntegrator(bdr_coef_g, 2, 1)); 
	gform.Assemble();
	gform *= 3.0;  

	mfem::ConstantCoefficient total_coef(total); 
	mfem::ConstantCoefficient scattering_coef(scattering); 
	mfem::ConstantCoefficient absorption_coef(total - scattering); 

	using HypreParMatrixPtr = std::unique_ptr<mfem::HypreParMatrix>; 
	mfem::ParBilinearForm Mtform(&vfes);
	mfem::ProductCoefficient total3(3.0, total_coef); 
	Mtform.AddDomainIntegrator(new mfem::VectorMassIntegrator(total3)); 
	Mtform.AddBdrFaceIntegrator(new DGVectorJumpJumpIntegrator(1./2/alpha)); 
	Mtform.Assemble(); 
	Mtform.Finalize();  
	auto Mt = HypreParMatrixPtr(Mtform.ParallelAssemble()); 
	auto iMt = HypreParMatrixPtr(ElementByElementBlockInverse(vfes, *Mt)); 

	mfem::ParBilinearForm Maform(&fes); 
	mfem::ConstantCoefficient alpha_c(alpha/2); 
	Maform.AddDomainIntegrator(new mfem::MassIntegrator(absorption_coef)); 
	Maform.AddInteriorFaceIntegrator(new PenaltyIntegrator(alpha/2, false)); 
	Maform.AddBdrFaceIntegrator(new mfem::BoundaryMassIntegrator(alpha_c)); 
	Maform.Assemble(); 
	Maform.Finalize(); 
	auto Ma = HypreParMatrixPtr(Maform.ParallelAssemble());

	mfem::ParMixedBilinearForm Dform(&vfes, &fes); 
	mfem::ConstantCoefficient neg_one(-1.0); 
	Dform.AddDomainIntegrator(new mfem::TransposeIntegrator(new mfem::GradientIntegrator(neg_one))); 
	mfem::Vector ldg_beta(dim); 
	for (auto d=0; d<dim; d++) { ldg_beta(d) = d+1; }
	Dform.AddInteriorFaceIntegrator(new mfem::LDGTraceIntegrator(&ldg_beta)); 
	Dform.AddBdrFaceIntegrator(new DGJumpAverageIntegrator(0.5)); 
	Dform.Assemble(); 
	Dform.Finalize(); 
	auto D = HypreParMatrixPtr(Dform.ParallelAssemble()); 
	auto DT = HypreParMatrixPtr(D->Transpose()); 

	auto DiMt = HypreParMatrixPtr(mfem::ParMult(D.get(), iMt.get(), true)); 
	auto DiMtDT = HypreParMatrixPtr(mfem::ParMult(DiMt.get(), DT.get(), true)); 
	auto S = HypreParMatrixPtr(mfem::ParAdd(DiMtDT.get(), Ma.get())); 

	DiMt->Mult(-1.0, gform, 1.0, fform); 

	mfem::CGSolver solver(MPI_COMM_WORLD); 
	solver.SetAbsTol(1e-10); 
	solver.SetMaxIter(100); 
	solver.SetPrintLevel(0); 
	mfem::HypreBoomerAMG amg(*S); 
	amg.SetPrintLevel(0); 
	solver.SetOperator(*S); 
	solver.SetPreconditioner(amg); 

	mfem::ParGridFunction phi(&fes), J(&vfes); 
	phi = 0.0; 
	solver.Mult(fform, phi); 

	DT->Mult(1.0, phi, 1.0, gform); 
	iMt->Mult(gform, J); 

	auto exsol = [&gamma, base](const mfem::Vector &x) {
		return sin(M_PI*x(0))*sin(M_PI*x(1)) + gamma*2./3*sin(3*M_PI*x(0))*sin(3*M_PI*x(1)) + base; 
	};
	mfem::FunctionCoefficient exsol_coef(exsol); 
	double err = phi.ComputeL2Error(exsol_coef); 

	auto Jexsol = [&delta](const mfem::Vector &x, mfem::Vector &v) {
		v = delta*sin(2*M_PI*x(0))*sin(2*M_PI*x(1))/3; 
	};
	mfem::VectorFunctionCoefficient Jexsol_coef(dim, Jexsol); 
	double Jerr = J.ComputeL2Error(Jexsol_coef); 

	return {err, Jerr}; 
}

TEST(SMM, IndependentLDGSMMp1) {
	auto [phi1, J1] = IndependentLDGSMMError(10, 1); 
	auto [phi2, J2] = IndependentLDGSMMError(20, 1); 
	double phi_ooa = log2(phi1/phi2); 
	double J_ooa = log2(J1/J2); 
	EXPECT_NEAR(phi_ooa, 2.0, 0.1); 
	EXPECT_NEAR(J_ooa, 2.0, 0.2); 
}

TEST(SMM, IndependentLDGSMMp2) {
	auto [phi1, J1] = IndependentLDGSMMError(10, 2); 
	auto [phi2, J2] = IndependentLDGSMMError(20, 2); 
	double phi_ooa = log2(phi1/phi2); 
	double J_ooa = log2(J1/J2); 
	EXPECT_NEAR(phi_ooa, 3.0, 0.15); 
	EXPECT_NEAR(J_ooa, 3.0, 0.15); 
}

TEST(SMM, IndependentLDGSMMp3) {
	auto [phi1, J1] = IndependentLDGSMMError(10, 3); 
	auto [phi2, J2] = IndependentLDGSMMError(20, 3); 
	double phi_ooa = log2(phi1/phi2); 
	double J_ooa = log2(J1/J2); 
	EXPECT_NEAR(phi_ooa, 4.0, 0.15); 
	EXPECT_NEAR(J_ooa, 4.0, 0.15); 
}