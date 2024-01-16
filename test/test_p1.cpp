#include "transport_op.hpp"
#include "sweep.hpp"
#include "p1diffusion.hpp"
#include "gtest/gtest.h"

TEST(Consistency, S21D) {
	mfem::Mesh smesh = mfem::Mesh::MakeCartesian1D(40, 1.0); 
	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(1, dim, mfem::BasisType::GaussLegendre); 
	mfem::ParFiniteElementSpace fes(&mesh, &fec); 
	mfem::ParFiniteElementSpace vfes(&mesh, &fec, dim);
	LevelSymmetricQuadrature quad(2, dim); 

	mfem::ConstantCoefficient total(1.0); 
	mfem::ConstantCoefficient scattering(.25); 
	mfem::SumCoefficient absorption(total, scattering, 1.0, -1.0); 
	mfem::ConstantCoefficient source(1.0); 
	mfem::ConstantCoefficient inflow(0.0); 

	TransportVectorExtents psi_ext(1, quad.Size(), fes.GetVSize()); 
	MomentVectorExtents phi_ext(1, 1, fes.GetVSize()); 
	InverseAdvectionOperator Linv(fes, quad, psi_ext, total, inflow); 
	DiscreteToMoment D(quad, psi_ext, phi_ext); 

	mfem::BilinearForm Ms_form(&fes); 
	Ms_form.AddDomainIntegrator(new mfem::MassIntegrator(scattering)); 
	Ms_form.Assemble(); 
	Ms_form.Finalize();

	mfem::Vector psi(TotalExtent(psi_ext)); 
	psi = 0.0; 
	TransportOperator T(D, Linv, Ms_form, psi); 

	auto qmms = [&quad](const mfem::Vector &x, const mfem::Vector &Omega) {
		return (Omega(0)*M_PI*cos(M_PI*x(0)) + .75*sin(M_PI*x(0)))/2; 
	};
	auto inflow_mms = [](const mfem::Vector &, const mfem::Vector &) {
		return 0.0; 
	};
	mfem::Vector source_vec; 
	FormTransportSource(fes, quad, psi_ext, qmms, inflow_mms, source_vec); 

	Linv.Mult(source_vec, psi); 
	mfem::Vector schur_source(TotalExtent(phi_ext)); 
	D.Mult(psi, schur_source); 

	mfem::GMRESSolver solver(MPI_COMM_WORLD); 
	solver.SetMaxIter(100); 
	solver.SetAbsTol(1e-10); 
	solver.SetPrintLevel(1); 
	solver.SetOperator(T); 
	mfem::ParGridFunction phi_sn(&fes); 
	phi_sn = 0.0; 
	solver.Mult(schur_source, phi_sn); 

	// mfem::Vector phi(phi_sn.Size()); 
	// Ms_form.Mult(phi_sn, phi); 
	// D.MultTranspose(phi, psi); 
	// psi += source_vec; 
	// Linv.Mult(psi, psi); 
	// D.Mult(psi, phi); 
	// mfem::Vector psi2(TotalExtent(psi_ext)); 
	// D.MultTranspose(phi, psi2); 
	// psi2 -= psi; 
	// std::cout << "non isotropy = " << psi2.Norml2() << std::endl; 

	mfem::Vector normal(dim); 
	normal = 0.0; normal(0) = 1.0; 
	double alpha = ComputeAlpha(quad, normal)/2;
	std::cout << "alpha = " << alpha << std::endl; 
	auto p1disc = std::unique_ptr<mfem::BlockOperator>(CreateP1DiffusionDiscretization(
		fes, vfes, total, absorption, alpha)); 
	auto mono = std::unique_ptr<mfem::HypreParMatrix>(BlockOperatorToMonolithic(*p1disc)); 
	mfem::SuperLURowLocMatrix slu_op(*mono); 
	mfem::SuperLUSolver slu(slu_op); 
	slu.SetPrintStatistics(false); 

	mfem::LinearForm fform(&fes);
	auto q0f = [](const mfem::Vector &x) {
		return .75*sin(M_PI*x(0)); 
	}; 
	mfem::FunctionCoefficient Q0(q0f);
	fform.AddDomainIntegrator(new mfem::DomainLFIntegrator(Q0)); 
	fform.Assemble(); 

	mfem::LinearForm gform(&vfes); 
	auto q1f = [&quad](const mfem::Vector &x) {
		return 1./3 * M_PI * cos(M_PI*x(0));  
	};
	mfem::FunctionCoefficient Q1(q1f); 
	gform.AddDomainIntegrator(new mfem::DomainLFIntegrator(Q1)); 
	gform.Assemble(); 
	gform *= 3.0; 

	mfem::BlockVector b(p1disc->RowOffsets()); 
	b.GetBlock(0) = gform; 
	b.GetBlock(1) = fform; 
	mfem::BlockVector x(p1disc->RowOffsets()); 
	x = 0.0; 

	slu.Mult(b, x); 
	mfem::ParGridFunction phi_p1(&fes); 
	phi_p1 = x.GetBlock(1); 

	mfem::GridFunctionCoefficient other_c(&phi_p1); 
	auto diff = phi_sn.ComputeL2Error(other_c); 
	std::cout << "diff = " << diff << std::endl; 
	// EXPECT_TRUE(diff < 1e-10); 

	auto exsol_f = [](const mfem::Vector &x) {
		return sin(M_PI*x(0)); 
	};
	mfem::FunctionCoefficient exsol(exsol_f); 
	double sn_err = phi_sn.ComputeL2Error(exsol); 
	double p1_err = phi_p1.ComputeL2Error(exsol); 
	printf("sn err = %.3e, p1 err = %.3e\n", sn_err, p1_err); 

	mfem::ParaViewDataCollection dc("compare", &mesh); 
	dc.RegisterField("sn", &phi_sn); 
	dc.RegisterField("p1", &phi_p1); 
	dc.Save(); 
}