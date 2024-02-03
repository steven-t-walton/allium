#include "gtest/gtest.h"
#include "cons_smm_op.hpp"
#include "smm_integrators.hpp"
#include "dg_trace_coll.hpp"
#include "transport_op.hpp"
#include "sweep.hpp"
#include "p1diffusion.hpp"
#include "smm_op.hpp"

TEST(CSMM, DGTraceColl) {
	auto mesh = mfem::Mesh::MakeCartesian2D(2,2, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	const auto dim = mesh.Dimension(); 
	auto fec = DGTrace_FECollection(1, dim); 
	auto fes = mfem::FiniteElementSpace(&mesh, &fec); 
	const auto *fe = fes.GetTraceElement(0, mesh.GetFaceGeometry(0)); 
	EXPECT_TRUE(fe); 
	EXPECT_EQ(fe->GetDof(), 2); 
}

class ProjectBetaTest : public testing::Test {
protected:
	mfem::Mesh smesh = mfem::Mesh::MakeCartesian2D(2,2,mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	mfem::ParMesh mesh = mfem::ParMesh(MPI_COMM_WORLD, smesh); 
	const int dim = mesh.Dimension();  

	DGTrace_FECollection tr_fec = DGTrace_FECollection(1, dim); 
	mfem::ParFiniteElementSpace tr_fes = mfem::ParFiniteElementSpace(&mesh, &tr_fec); 
	mfem::ParFiniteElementSpace tr_vfes = mfem::ParFiniteElementSpace(&mesh, &tr_fec, dim); 
	mfem::ParGridFunction beta = mfem::ParGridFunction(&tr_fes);
	mfem::ParGridFunction tensor = mfem::ParGridFunction(&tr_vfes); 

	void ProjectClosures(PhaseSpaceCoefficient &f, int sn_order) {
		auto fec = mfem::L2_FECollection(1, dim); 
		auto fes = mfem::ParFiniteElementSpace(&mesh, &fec); 
		auto vfes = mfem::ParFiniteElementSpace(&mesh, &fec, dim); 
		auto quad = LevelSymmetricQuadrature(sn_order, dim); 
		auto psi_ext = TransportVectorExtents(1,quad.Size(), fes.GetVSize()); 
		auto psi = mfem::Vector(TotalExtent(psi_ext)); 
		auto psi_view = TransportVectorView(psi.GetData(), psi_ext); 
		mfem::Vector nor(dim); 
		nor = 0.0; 
		nor(0) = 1.0; 
		const auto alpha = ComputeAlpha(quad, nor);
		ProjectPsi(fes, quad, f, psi_view); 
		ProjectClosuresToFaces(fes, quad, psi_view, alpha, beta, tensor); 
	}
};

TEST_F(ProjectBetaTest, Isotropic) {
	ConstantPhaseSpaceCoefficient f(1.0);  
	ProjectClosures(f, 4); 
	EXPECT_NEAR(beta.Norml2(), 0.0, 1e-13); 
}

TEST_F(ProjectBetaTest, SpatialIsotropic) {
	auto iso = [](const mfem::Vector &x, const mfem::Vector &Omega) {
		return sin(M_PI*x(0))*sin(M_PI*x(1))/4/M_PI; 
	};
	FunctionGrayCoefficient iso_coef(iso); 
	ProjectClosures(iso_coef, 4); 
	EXPECT_NEAR(beta.Norml2(), 0.0, 1e-13); 
}

TEST_F(ProjectBetaTest, Quadratic) {
	auto angular = [](const mfem::Vector &x, const mfem::Vector &Omega) {
		return Omega*Omega/4/M_PI; 
	};
	FunctionGrayCoefficient angular_coef(angular); 
	ProjectClosures(angular_coef, 8); 
	beta -= 0.041666666666666664; 
	double E1 = beta.Norml2(); 
	ProjectClosures(angular_coef, 16); 
	beta -= 0.041666666666666664; 
	double E2 = beta.Norml2(); 
	int N1 = 40, N2 = 144; 
	auto ooa = log(E1/E2)/log(N2/N1); 
	EXPECT_NEAR(ooa, 1.0, 0.2); 
}

TEST(CSMM, ZerothLFI) {
	auto smesh = mfem::Mesh::MakeCartesian2D(2,2, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false); 
	auto mesh = mfem::ParMesh(MPI_COMM_WORLD, smesh); 
	const auto dim = mesh.Dimension(); 
	auto fec = mfem::L2_FECollection(1, dim, mfem::BasisType::GaussLobatto); 
	auto fes = mfem::ParFiniteElementSpace(&mesh, &fec); 
	auto vfes = mfem::ParFiniteElementSpace(&mesh, &fec, dim); 
	LevelSymmetricQuadrature quad(8, dim); 
	TransportVectorExtents psi_ext(1,quad.Size(),fes.GetVSize()); 
	const auto psi_size = TotalExtent(psi_ext); 
	mfem::Vector psi(psi_size), source(psi_size); 
	TransportVectorView psi_view(psi.GetData(), psi_ext), source_view(source.GetData(), psi_ext); 
	psi = 1.0; 
	source = 0.0; 
	mfem::Vector nor(dim); 
	nor = 0.0; 
	nor(0) = 1.0; 
	double alpha = ComputeAlpha(quad, nor);

	auto tr_fec = DGTrace_FECollection(1, dim); 
	auto tr_fes = mfem::ParFiniteElementSpace(&mesh, &tr_fec); 
	auto tr_vfes = mfem::ParFiniteElementSpace(&mesh, &tr_fec, dim); 
	mfem::ParGridFunction beta(&tr_fes), tensor(&tr_vfes); 
	auto angle_space = [](const mfem::Vector &x, const mfem::Vector &Omega) {
		return (sin(M_PI*x(0))*sin(M_PI*x(1)) + (Omega(0) + Omega(1))*sin(2*M_PI*x(0))*sin(2*M_PI*x(1)) + (Omega*Omega)*sin(3*M_PI*(x(0)+.05)/1.1)*sin(3*M_PI*(x(1)+.05)/1.1))/4/M_PI; 
	};
	FunctionGrayCoefficient angle_space_coef(angle_space); 
	ProjectPsi(fes, quad, angle_space_coef, psi_view); 
	ProjectClosuresToFaces(fes, quad, psi_view, alpha, beta, tensor); 

	mfem::ParLinearForm fform(&fes); 
	fform.AddInteriorFaceIntegrator(new CSMMZerothMomentFaceLFIntegrator(beta));
	fform.Assemble(); 
	// [[ beta ]] = 0 when GaussLobatto used 
	EXPECT_NEAR(fform.Norml2(), 0.0, 1e-15); 
}

#ifdef MFEM_USE_SUPERLU
std::tuple<double,double> P1SMMError(int Ne, int fe_order) {
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
	psi = 0.0; 
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
	ConsistentSMMSourceOperator source_op(fes, vfes, quad, psi_ext, source_view, alpha); 
	auto *p1 = CreateP1DiffusionDiscretization(fes, vfes, total_coef, absorption_coef, alpha); 
	auto *mono = BlockOperatorToMonolithic(*p1); 
	mfem::SuperLURowLocMatrix slu_op(*mono); 
	mfem::SuperLUSolver slu(slu_op); 
	slu.SetPrintStatistics(false); 
	mfem::ProductOperator smm(&slu, &source_op, false, false); 

	mfem::BlockVector x(p1->RowOffsets()), xold(p1->RowOffsets()); 
	x = 0.0; 
	xold = 0.0; 
	mfem::ParGridFunction phi(&fes, x.GetBlock(1)), J(&vfes, x.GetBlock(0)); 
	mfem::Vector scat_source(phi_size); 
	int it; 
	double norm; 
	for (it=0; it<10; it++) {
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

	mfem::ParGridFunction phi_sn(&fes); 
	D.Mult(psi, phi_sn); 
	mfem::GridFunctionCoefficient phi_sn_c(&phi_sn); 
	double consistency = phi.ComputeL2Error(phi_sn_c); 
	EXPECT_NEAR(consistency, 0.0, 1e-12); 

	// compute errors 
	mfem::FunctionCoefficient exsol_coef(phi_func); 
	double err = phi.ComputeL2Error(exsol_coef); 

	mfem::VectorFunctionCoefficient Jexsol_coef(dim, J_func); 
	double Jerr = J.ComputeL2Error(Jexsol_coef);

	delete p1; 
	delete mono; 

	return {err, Jerr}; 
}

TEST(P1SMM, MMSp1) {
	auto [phi1, J1] = P1SMMError(10, 1); 
	auto [phi2, J2] = P1SMMError(20, 1); 
	double phi_ooa = log2(phi1/phi2); 
	double J_ooa = log2(J1/J2); 
	EXPECT_NEAR(phi_ooa, 2.0, 0.1); 
	EXPECT_NEAR(J_ooa, 2.0, 0.2); 
}

TEST(P1SMM, MMSp2) {
	auto [phi1, J1] = P1SMMError(10, 2); 
	auto [phi2, J2] = P1SMMError(20, 2); 
	double phi_ooa = log2(phi1/phi2); 
	double J_ooa = log2(J1/J2); 
	EXPECT_NEAR(phi_ooa, 3.0, 0.1); 
	EXPECT_NEAR(J_ooa, 3.0, 0.2); 
}
#endif

std::tuple<double,double> LDGSMMError(int Ne, int fe_order) {
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
	psi = 0.0; 
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

	mfem::Vector beta(dim); 
	for (auto d=0; d<dim; d++) { beta(d) = d+1; }
	ConsistentLDGSMMSourceOperator source_op(fes, vfes, quad, psi_ext, source_view, alpha, beta);
	LDGDiffusionDiscretization ldg(fes, vfes, total_coef, absorption_coef, alpha, beta); 
	const auto &S = ldg.SchurComplement(); 

	mfem::CGSolver solver(MPI_COMM_WORLD); 
	solver.SetAbsTol(1e-12); 
	solver.SetMaxIter(100); 
	solver.SetPrintLevel(0); 
	mfem::HypreBoomerAMG amg(S); 
	amg.SetPrintLevel(0); 
	solver.SetOperator(S); 
	solver.SetPreconditioner(amg); 
	solver.iterative_mode = false; 
	InverseBlockDiffusionOperator inv_ldg(ldg, solver); 
	mfem::ProductOperator smm(&inv_ldg, &source_op, false, false); 

	mfem::BlockVector x(ldg.GetOffsets()), xold(ldg.GetOffsets()); 
	x = 0.0; 
	xold = 0.0; 
	mfem::ParGridFunction phi(&fes, x.GetBlock(1)), J(&vfes, x.GetBlock(0)); 
	mfem::Vector scat_source(phi_size); 
	int it; 
	double norm; 
	for (it=0; it<10; it++) {
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

	mfem::ParGridFunction phi_sn(&fes); 
	D.Mult(psi, phi_sn); 
	mfem::GridFunctionCoefficient phi_sn_c(&phi_sn); 
	double consistency = phi.ComputeL2Error(phi_sn_c); 
	EXPECT_NEAR(consistency, 0.0, 1e-12); 

	// compute errors 
	mfem::FunctionCoefficient exsol_coef(phi_func); 
	double err = phi.ComputeL2Error(exsol_coef); 

	mfem::VectorFunctionCoefficient Jexsol_coef(dim, J_func); 
	double Jerr = J.ComputeL2Error(Jexsol_coef);

	return {err, Jerr}; 
}

TEST(ConsLDGSMM, MMSp1) {
	auto [phi1, J1] = LDGSMMError(10, 1); 
	auto [phi2, J2] = LDGSMMError(20, 1); 
	double phi_ooa = log2(phi1/phi2); 
	double J_ooa = log2(J1/J2); 
	EXPECT_NEAR(phi_ooa, 2.0, 0.1); 
	EXPECT_NEAR(J_ooa, 2.0, 0.2); 
}

TEST(ConsLDGSMM, MMSp2) {
	auto [phi1, J1] = LDGSMMError(10, 2); 
	auto [phi2, J2] = LDGSMMError(20, 2); 
	double phi_ooa = log2(phi1/phi2); 
	double J_ooa = log2(J1/J2); 
	EXPECT_NEAR(phi_ooa, 3.0, 0.1); 
	EXPECT_NEAR(J_ooa, 3.0, 0.2); 
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
	psi = 0.0; 
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

	mfem::Vector beta(dim); 
	for (auto d=0; d<dim; d++) { beta(d) = d+1; }
	ConsistentIPSMMSourceOperator source_op(fes, vfes, quad, psi_ext, source_view, alpha, total_coef);
	IPDiffusionDiscretization ip(fes, vfes, total_coef, absorption_coef, alpha); 
	const auto &S = ip.SchurComplement(); 

	mfem::CGSolver solver(MPI_COMM_WORLD); 
	solver.SetAbsTol(1e-12); 
	solver.SetMaxIter(100); 
	solver.SetPrintLevel(0); 
	mfem::HypreBoomerAMG amg(S); 
	amg.SetPrintLevel(0); 
	solver.SetOperator(S); 
	solver.SetPreconditioner(amg); 
	solver.iterative_mode = false; 
	InverseBlockDiffusionOperator inv_ip(ip, solver); 
	mfem::ProductOperator smm(&inv_ip, &source_op, false, false); 

	mfem::BlockVector x(ip.GetOffsets()), xold(ip.GetOffsets()); 
	x = 0.0; 
	xold = 0.0; 
	mfem::ParGridFunction phi(&fes, x.GetBlock(1)), J(&vfes, x.GetBlock(0)); 
	mfem::Vector scat_source(phi_size); 
	int it; 
	double norm; 
	for (it=0; it<10; it++) {
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

	mfem::ParGridFunction phi_sn(&fes); 
	D.Mult(psi, phi_sn); 
	mfem::GridFunctionCoefficient phi_sn_c(&phi_sn); 
	double consistency = phi.ComputeL2Error(phi_sn_c); 
	EXPECT_NEAR(consistency, 0.0, 1e-11); 

	// compute errors 
	mfem::FunctionCoefficient exsol_coef(phi_func); 
	double err = phi.ComputeL2Error(exsol_coef); 

	mfem::VectorFunctionCoefficient Jexsol_coef(dim, J_func); 
	double Jerr = J.ComputeL2Error(Jexsol_coef);

	return {err, Jerr}; 
}

TEST(ConsIPSMM, MMSp1) {
	auto [phi1, J1] = IPSMMError(10, 1); 
	auto [phi2, J2] = IPSMMError(20, 1); 
	double phi_ooa = log2(phi1/phi2); 
	double J_ooa = log2(J1/J2); 
	EXPECT_NEAR(phi_ooa, 2.0, 0.1); 
	EXPECT_NEAR(J_ooa, 2.0, 0.2); 	
}

TEST(ConsIPSMM, MMSp2) {
	auto [phi1, J1] = IPSMMError(10, 2); 
	auto [phi2, J2] = IPSMMError(20, 2); 
	double phi_ooa = log2(phi1/phi2); 
	double J_ooa = log2(J1/J2); 
	EXPECT_NEAR(phi_ooa, 3.0, 0.1); 
	EXPECT_NEAR(J_ooa, 3.0, 0.2); 	
}

TEST(ConsIPSMM, MMSp3) {
	auto [phi1, J1] = IPSMMError(10, 3); 
	auto [phi2, J2] = IPSMMError(20, 3); 
	double phi_ooa = log2(phi1/phi2); 
	double J_ooa = log2(J1/J2); 
	EXPECT_NEAR(phi_ooa, 4.0, 0.1); 
	EXPECT_NEAR(J_ooa, 4.0, 0.2); 	
}