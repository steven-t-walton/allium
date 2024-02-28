#include "gtest/gtest.h"
#include "p1diffusion.hpp"
#include "smm_integrators.hpp"
#include "phase_coefficient.hpp"
#include "linalg.hpp"
#include "block_smm_op.hpp"
#include "sweep.hpp"
#include "transport_op.hpp"

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
	const auto psi_size = TotalExtent(psi_ext); 
	mfem::Vector psi(psi_size), source_vec(psi_size); 
	TransportVectorView psi_view(psi.GetData(), psi_ext), source_view(source_vec.GetData(), psi_ext); 

	// exact solutions 
	auto f = [&delta, &gamma, base](const mfem::Vector &x, const mfem::Vector &Omega) {
		return (sin(M_PI*x(0))*sin(M_PI*x(1)) + delta*(Omega(0) + Omega(1))*sin(2*M_PI*x(0))*sin(2*M_PI*x(1)) 
			+ gamma*(Omega*Omega)*sin(3*M_PI*x(0))*sin(3*M_PI*x(1)) + base)/4/M_PI; 
	};
	auto phi_func = [gamma, base](const mfem::Vector &x) {
		return sin(M_PI*x(0))*sin(M_PI*x(1)) + gamma*2./3*sin(3*M_PI*x(0))*sin(3*M_PI*x(1)) + base; 
	};
	auto J_func = [&delta](const mfem::Vector &x, mfem::Vector &v) {
		v = delta*sin(2*M_PI*x(0))*sin(2*M_PI*x(1))/3; 
	};

	// project onto discrete psi 
	FunctionGrayCoefficient psi_coef(f); 

	// mms source 
	auto q = [delta, gamma, total, scattering, &f, &phi_func](const mfem::Vector &X, const mfem::Vector &Omega) {
		double x = M_PI*X(0);
		double y = M_PI*X(1); 
		double mu = Omega(0); 
		double eta = Omega(1); 
		double dpsidx = .25*(cos(x)*sin(y) + 2*delta*(mu+eta)*cos(2*x)*sin(2*y) + 3*gamma*(Omega*Omega)*cos(3*x)*sin(3*y)); 
		double dpsidy = .25*(sin(x)*cos(y) + 2*delta*(mu+eta)*sin(2*x)*cos(2*y) + 3*gamma*(Omega*Omega)*sin(3*x)*cos(3*y)); 
		return mu*dpsidx + eta*dpsidy + total*f(X,Omega) - scattering*phi_func(X)/4/M_PI; 
	};
	FunctionGrayCoefficient source_coef(q); 

	// compute alpha coefficient 
	mfem::Vector nor(dim); 
	nor = 0.0; 
	nor(0) = 1.0; 
	const auto alpha = ComputeAlpha(quad, nor); 

	mfem::ConstantCoefficient total_coef(total); 
	mfem::ConstantCoefficient scattering_coef(scattering); 
	mfem::ConstantCoefficient absorption_coef(total - scattering); 

	// setup SMM operators 
	BlockDiffusionSMMSourceOperator source_op(fes, vfes, quad, psi_ext, source_coef, psi_coef, alpha); 
	mfem::Vector beta(dim); 
	for (auto d=0; d<dim; d++) { beta(d) = d+1; }
	BlockLDGDiffusionDiscretization ldg(fes, vfes, total_coef, absorption_coef, alpha, beta); 
	const auto &S = ldg.SchurComplement(); 

	// solve schur complement 
	mfem::CGSolver solver(MPI_COMM_WORLD); 
	solver.SetAbsTol(1e-12); 
	solver.SetMaxIter(100); 
	solver.SetPrintLevel(0); 
	mfem::HypreBoomerAMG amg(S); 
	amg.SetPrintLevel(0); 
	solver.SetOperator(S); 
	solver.SetPreconditioner(amg); 
	solver.iterative_mode = false; 

	InverseAdvectionOperator Linv(fes, quad, psi_ext, total_coef, psi_coef); 
	mfem::BilinearForm Ms_form(&fes); 
	Ms_form.AddDomainIntegrator(new mfem::MassIntegrator(scattering_coef)); 
	Ms_form.Assemble(); 
	Ms_form.Finalize();
	MomentVectorExtents phi_ext(1, 1, fes.GetVSize());
	const auto phi_size = TotalExtent(phi_ext); 
	// integrates over angle psi -> phi 
	DiscreteToMoment D(quad, psi_ext, phi_ext); 
	FormTransportSource(fes, quad, source_coef, psi_coef, source_view); 

	InverseBlockDiffusionOperator inv_ldg(ldg, solver); 
	mfem::ProductOperator smm(&inv_ldg, &source_op, false, false); 

	mfem::BlockVector x(ldg.GetOffsets()), xold(ldg.GetOffsets()); 
	x = 0.0; 
	xold = 0.0; 
	mfem::ParGridFunction phi(&fes, x.GetBlock(1)), J(&vfes, x.GetBlock(0)); 
	mfem::Vector scat_source(phi_size); 
	int it; 
	double norm; 
	for (it=0; it<50; it++) {
		Ms_form.Mult(xold.GetBlock(1), scat_source); 
		D.MultTranspose(scat_source, psi); 
		psi += source_vec; 
		Linv.Mult(psi, psi); 
		smm.Mult(psi, x); 

		xold -= x; 
		norm = sqrt(mfem::InnerProduct(MPI_COMM_WORLD, xold, xold)); 
		xold = x; 
		if (norm < 1e-10) break; 
	}
	EXPECT_TRUE(it < 15); 
	EXPECT_TRUE(norm < 1e-10); 

	// compute errors 
	mfem::FunctionCoefficient exsol_coef(phi_func); 
	double err = phi.ComputeL2Error(exsol_coef); 

	mfem::VectorFunctionCoefficient Jexsol_coef(dim, J_func); 
	double Jerr = J.ComputeL2Error(Jexsol_coef); 

	return {err, Jerr}; 
}

TEST(LDGSMM, MMSp1) {
	auto [phi1, J1] = IndependentLDGSMMError(10, 1); 
	auto [phi2, J2] = IndependentLDGSMMError(20, 1); 
	double phi_ooa = log2(phi1/phi2); 
	double J_ooa = log2(J1/J2); 
	EXPECT_NEAR(phi_ooa, 2.0, 0.1); 
	EXPECT_NEAR(J_ooa, 2.0, 0.2); 
}

TEST(LDGSMM, MMSp2) {
	auto [phi1, J1] = IndependentLDGSMMError(20, 2); 
	auto [phi2, J2] = IndependentLDGSMMError(40, 2); 
	double phi_ooa = log2(phi1/phi2); 
	double J_ooa = log2(J1/J2); 
	EXPECT_NEAR(phi_ooa, 3.0, 0.15); 
	EXPECT_NEAR(J_ooa, 2.0, 0.3); // <-- losing an order in J for even p only?  
}

TEST(LDGSMM, MMSp3) {
	auto [phi1, J1] = IndependentLDGSMMError(10, 3); 
	auto [phi2, J2] = IndependentLDGSMMError(20, 3); 
	double phi_ooa = log2(phi1/phi2); 
	double J_ooa = log2(J1/J2); 
	EXPECT_NEAR(phi_ooa, 4.0, 0.15); 
	EXPECT_NEAR(J_ooa, 4.0, 0.15); 
}

std::tuple<double,double> IPSMMError(int Ne, int fe_order) {
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
	const auto psi_size = TotalExtent(psi_ext); 
	mfem::Vector psi(psi_size), source_vec(psi_size); 
	TransportVectorView psi_view(psi.GetData(), psi_ext), source_view(source_vec.GetData(), psi_ext); 

	// exact solutions 
	auto f = [&delta, &gamma, base](const mfem::Vector &x, const mfem::Vector &Omega) {
		return (sin(M_PI*x(0))*sin(M_PI*x(1)) + delta*(Omega(0) + Omega(1))*sin(2*M_PI*x(0))*sin(2*M_PI*x(1)) 
			+ gamma*(Omega*Omega)*sin(3*M_PI*x(0))*sin(3*M_PI*x(1)) + base)/4/M_PI; 
	};
	auto phi_func = [gamma, base](const mfem::Vector &x) {
		return sin(M_PI*x(0))*sin(M_PI*x(1)) + gamma*2./3*sin(3*M_PI*x(0))*sin(3*M_PI*x(1)) + base; 
	};
	auto J_func = [&delta](const mfem::Vector &x, mfem::Vector &v) {
		v = delta*sin(2*M_PI*x(0))*sin(2*M_PI*x(1))/3; 
	};

	// project onto discrete psi 
	FunctionGrayCoefficient psi_coef(f); 

	// mms source 
	auto q = [delta, gamma, total, scattering, &f, &phi_func](const mfem::Vector &X, const mfem::Vector &Omega) {
		double x = M_PI*X(0);
		double y = M_PI*X(1); 
		double mu = Omega(0); 
		double eta = Omega(1); 
		double dpsidx = .25*(cos(x)*sin(y) + 2*delta*(mu+eta)*cos(2*x)*sin(2*y) + 3*gamma*(Omega*Omega)*cos(3*x)*sin(3*y)); 
		double dpsidy = .25*(sin(x)*cos(y) + 2*delta*(mu+eta)*sin(2*x)*cos(2*y) + 3*gamma*(Omega*Omega)*sin(3*x)*cos(3*y)); 
		return mu*dpsidx + eta*dpsidy + total*f(X,Omega) - scattering*phi_func(X)/4/M_PI; 
	};
	FunctionGrayCoefficient source_coef(q); 

	// compute alpha coefficient 
	mfem::Vector nor(dim); 
	nor = 0.0; 
	nor(0) = 1.0; 
	const auto alpha = ComputeAlpha(quad, nor); 

	mfem::ConstantCoefficient total_coef(total); 
	mfem::ConstantCoefficient scattering_coef(scattering); 
	mfem::ConstantCoefficient absorption_coef(total - scattering); 

	// setup SMM operators 
	BlockDiffusionSMMSourceOperator source_op(fes, vfes, quad, psi_ext, source_coef, psi_coef, alpha); 
	mfem::Vector beta(dim); 
	for (auto d=0; d<dim; d++) { beta(d) = d+1; }
	BlockIPDiffusionDiscretization ip(fes, vfes, total_coef, absorption_coef, alpha); 
	const auto &S = ip.SchurComplement(); 

	// solve schur complement 
	mfem::CGSolver solver(MPI_COMM_WORLD); 
	solver.SetAbsTol(1e-12); 
	solver.SetMaxIter(100); 
	solver.SetPrintLevel(0); 
	mfem::HypreBoomerAMG amg(S); 
	amg.SetPrintLevel(0); 
	solver.SetOperator(S); 
	solver.SetPreconditioner(amg); 
	solver.iterative_mode = false; 

	InverseAdvectionOperator Linv(fes, quad, psi_ext, total_coef, psi_coef); 
	mfem::BilinearForm Ms_form(&fes); 
	Ms_form.AddDomainIntegrator(new mfem::MassIntegrator(scattering_coef)); 
	Ms_form.Assemble(); 
	Ms_form.Finalize();
	MomentVectorExtents phi_ext(1, 1, fes.GetVSize());
	const auto phi_size = TotalExtent(phi_ext); 
	// integrates over angle psi -> phi 
	DiscreteToMoment D(quad, psi_ext, phi_ext); 
	FormTransportSource(fes, quad, source_coef, psi_coef, source_view); 

	InverseBlockDiffusionOperator inv_ip(ip, solver); 
	mfem::ProductOperator smm(&inv_ip, &source_op, false, false); 

	mfem::BlockVector x(ip.GetOffsets()), xold(ip.GetOffsets()); 
	x = 0.0; 
	xold = 0.0; 
	mfem::ParGridFunction phi(&fes, x.GetBlock(1)), J(&vfes, x.GetBlock(0)); 
	mfem::Vector scat_source(phi_size); 
	int it; 
	double norm; 
	for (it=0; it<50; it++) {
		Ms_form.Mult(xold.GetBlock(1), scat_source); 
		D.MultTranspose(scat_source, psi); 
		psi += source_vec; 
		Linv.Mult(psi, psi); 
		smm.Mult(psi, x); 

		xold -= x; 
		norm = sqrt(mfem::InnerProduct(MPI_COMM_WORLD, xold, xold)); 
		xold = x; 
		if (norm < 1e-10) break; 
	}
	EXPECT_TRUE(it < 15); 
	EXPECT_TRUE(norm < 1e-10); 

	// compute errors 
	mfem::FunctionCoefficient exsol_coef(phi_func); 
	double err = phi.ComputeL2Error(exsol_coef); 

	mfem::VectorFunctionCoefficient Jexsol_coef(dim, J_func); 
	double Jerr = J.ComputeL2Error(Jexsol_coef); 

	return {err, Jerr}; 
}

TEST(IPSMM, MMSp1) {
	auto [phi1, J1] = IPSMMError(20, 1); 
	auto [phi2, J2] = IPSMMError(40, 1); 
	double phi_ooa = log2(phi1/phi2); 
	double J_ooa = log2(J1/J2); 
	EXPECT_NEAR(phi_ooa, 2.0, 0.15); 
	EXPECT_NEAR(J_ooa, 1.0, 0.2); // <-- IP loses an order for vector variable 	
}

TEST(IPSMM, MMSp2) {
	auto [phi1, J1] = IPSMMError(10, 2); 
	auto [phi2, J2] = IPSMMError(20, 2); 
	double phi_ooa = log2(phi1/phi2); 
	double J_ooa = log2(J1/J2); 
	EXPECT_NEAR(phi_ooa, 3.0, 0.15); 
	EXPECT_NEAR(J_ooa, 2.0, 0.15); // <-- IP loses an order for vector variable 	
}

TEST(IPSMM, MMSp3) {
	auto [phi1, J1] = IPSMMError(15, 3); 
	auto [phi2, J2] = IPSMMError(30, 3); 
	double phi_ooa = log2(phi1/phi2); 
	double J_ooa = log2(J1/J2); 
	EXPECT_NEAR(phi_ooa, 4.0, 0.15); 
	EXPECT_NEAR(J_ooa, 3.0, 0.15); // <-- IP loses an order for vector variable 	
}