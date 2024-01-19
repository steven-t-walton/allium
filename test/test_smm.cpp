#include "gtest/gtest.h"
#include "smm_integrators.hpp"
#include "p1diffusion.hpp"

void ProjectPsi(mfem::FiniteElementSpace &fes, AngularQuadrature &quad, 
	std::function<double(const mfem::Vector &x, const mfem::Vector &Omega)> f, TransportVectorView psi) 
{
	mfem::Array<int> dofs; 
	mfem::Vector vals; 
	for (auto a=0; a<quad.Size(); a++) {
		const auto &Omega = quad.GetOmega(a); 
		auto f_omega = [&Omega, &f](const mfem::Vector &x) {
			return f(x,Omega); 
		};
		mfem::FunctionCoefficient coef(f_omega); 
		for (auto e=0; e<fes.GetNE(); e++) {
			fes.GetElementDofs(e, dofs); 
			vals.SetSize(dofs.Size()); 
			fes.GetFE(e)->Project(coef, *fes.GetElementTransformation(e), vals); 
			for (auto i=0; i<vals.Size(); i++) {
				psi(0,a,dofs[i]) = vals[i]; 
			}
		}
	}
}

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
	TransportVectorView psi_view(psi.GetData(), psi_ext); 
	ProjectPsi(fes, quad, f, psi_view); 
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
	TransportVectorView psi_view(psi.GetData(), psi_ext); 
	ProjectPsi(fes, quad, f, psi_view); 

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
	TransportVectorView psi_view(psi.GetData(), psi_ext); 
	ProjectPsi(fes, quad, f, psi_view); 

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
	TransportVectorView psi_view(psi.GetData(), psi_ext); 
	ProjectPsi(fes, quad, f, psi_view); 

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

TEST(SMM, MMS) {
	auto Ne = 40; 
	auto fe_order = 1; 
	auto sn_order = 8; 
	mfem::Mesh smesh = mfem::Mesh::MakeCartesian2D(Ne, Ne, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh); 
	const auto dim = mesh.Dimension(); 

	double total = 1.0; 
	double scattering = .25; 
	double gamma = 0.0; 
	
	mfem::L2_FECollection fec(fe_order, dim, mfem::BasisType::GaussLegendre); 
	mfem::ParFiniteElementSpace fes(&mesh, &fec); 
	mfem::ParFiniteElementSpace vfes(&mesh, &fec, dim); 
	LevelSymmetricQuadrature quad(sn_order, dim); 

	TransportVectorExtents psi_ext(1,quad.Size(),fes.GetVSize()); 
	mfem::Vector psi(TotalExtent(psi_ext)); 
	auto f = [&gamma](const mfem::Vector &x, const mfem::Vector &Omega) {
		return (sin(M_PI*x(0))*sin(M_PI*x(1)) + gamma*(Omega*Omega)*sin(2*M_PI*x(0))*sin(2*M_PI*x(1)))/4/M_PI; 
	};
	TransportVectorView psi_view(psi.GetData(), psi_ext); 
	ProjectPsi(fes, quad, f, psi_view); 

	mfem::Vector nor(dim); 
	nor = 0.0; 
	nor(0) = 1.0; 
	auto alpha = ComputeAlpha(quad, nor); 
	SMMCorrectionTensorCoefficient T(fes, quad, psi_view); 
	SMMBdrCorrectionFactorCoefficient beta(fes, quad, psi_view, alpha); 	

	auto Q0f = [&total, &scattering, &gamma](const mfem::Vector &x) {
		return (total-scattering)*(sin(M_PI*x(0))*sin(M_PI*x(1)) + gamma*2./3*sin(2*M_PI*x(0))*sin(2*M_PI*x(1))); 
	};
	mfem::FunctionCoefficient Q0(Q0f); 
	auto Q1f = [&gamma](const mfem::Vector &x, mfem::Vector &v) {
		v.SetSize(x.Size()); 
		v(0) = M_PI/15*(5*cos(M_PI*x(0))*sin(M_PI*x(1)) + gamma*8*cos(2*M_PI*x(0))*sin(2*M_PI*x(1))); 
		v(1) = M_PI/15*(5*sin(M_PI*x(0))*cos(M_PI*x(1)) + gamma*8*sin(2*M_PI*x(0))*cos(2*M_PI*x(1))); 
	};
	mfem::VectorFunctionCoefficient Q1(dim, Q1f); 

	mfem::ParLinearForm fform(&fes); 
	fform.AddDomainIntegrator(new mfem::DomainLFIntegrator(Q0)); 
	fform.Assemble(); 

	mfem::ParLinearForm gform(&vfes); 
	gform.AddDomainIntegrator(new mfem::VectorDomainLFIntegrator(Q1)); 
	gform.AddDomainIntegrator(new WeakTensorDivergenceLFIntegrator(T)); 
	gform.AddInteriorFaceIntegrator(new VectorJumpTensorAverageLFIntegrator(T)); 
	gform.AddBdrFaceIntegrator(new VectorJumpTensorAverageLFIntegrator(T)); 
	mfem::ProductCoefficient bdr_coef(1./(3*alpha), beta); 
	gform.AddBdrFaceIntegrator(new BoundaryNormalFaceLFIntegrator(bdr_coef)); 
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

	mfem::ParBilinearForm Maform(&fes); 
	mfem::ConstantCoefficient alpha_c(alpha); 
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
	// Dform.AddBdrFaceIntegrator(new mfem::LDGTraceIntegrator); 
	Dform.Assemble(); 
	Dform.Finalize(); 
	auto D = HypreParMatrixPtr(Dform.ParallelAssemble()); 
	auto DT = HypreParMatrixPtr(D->Transpose()); 
	(*DT) *= -1.0; 

	mfem::Array<int> offsets(3); 
	offsets[0] = 0; 
	offsets[1] = vfes.GetVSize(); 
	offsets[2] = fes.GetVSize(); 
	offsets.PartialSum();
	mfem::BlockOperator bop(offsets); 
	bop.SetBlock(0,0, Mt.get()); 
	bop.SetBlock(0,1, DT.get()); 
	bop.SetBlock(1,0, D.get()); 
	bop.SetBlock(1,1, Ma.get()); 

	auto mono = HypreParMatrixPtr(BlockOperatorToMonolithic(bop)); 
	mfem::SuperLURowLocMatrix slu_op(*mono); 
	mfem::SuperLUSolver slu(slu_op); 
	slu.SetPrintStatistics(false); 

	mfem::BlockVector b(offsets), x(offsets); 
	b.GetBlock(0) = gform; 
	b.GetBlock(1) = fform; 
	slu.Mult(b, x); 

	mfem::ParGridFunction phi(&fes, x.GetBlock(1)); 
	auto exsol = [&gamma](const mfem::Vector &x) {
		return sin(M_PI*x(0))*sin(M_PI*x(1)) + gamma*2./3*sin(2*M_PI*x(0))*sin(2*M_PI*x(1)); 
	};
	mfem::FunctionCoefficient exsol_coef(exsol); 
	double err = phi.ComputeL2Error(exsol_coef); 
	printf("err = %.3e\n", err); 

	mfem::ParaViewDataCollection dc("solution", &mesh); 
	dc.RegisterField("phi", &phi); 
	dc.Save(); 	
}