#include "gtest/gtest.h"
#include "sweep.hpp"
#include "transport_op.hpp"

double LinearTransportError(mfem::Mesh &smesh, int fe_order) {
	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh); 
	const auto dim = mesh.Dimension();
	LevelSymmetricQuadrature quad(4, dim); 

	mfem::L2_FECollection fec(fe_order, dim, mfem::BasisType::GaussLegendre); 
	mfem::ParFiniteElementSpace fes(&mesh, &fec); 

	double total_val = 1.0; 
	double scattering_val = .25; 
	mfem::ConstantCoefficient total(total_val);
	mfem::ConstantCoefficient scattering(scattering_val);  
	mfem::ConstantCoefficient inflow(2.0/4/M_PI); 

	TransportVectorExtents psi_ext(1, quad.Size(), fes.GetVSize()); 
	MomentVectorExtents phi_ext(1, 1, fes.GetVSize()); 
	const auto psi_size = TotalExtent(psi_ext); 
	mfem::Vector source(psi_size), psi(psi_size); 
	psi = 0.0; 

	DiscreteToMoment D(quad, psi_ext, phi_ext); 

	TransportVectorView source_view(source.GetData(), psi_ext); 

	auto qmms = [&total_val, &scattering_val](const mfem::Vector &x, const mfem::Vector &Omega) {
		double quadratic = Omega*Omega; 
		auto dpsi_dx = cos(M_PI*x(0))*sin(M_PI*x(1))/4 + quadratic*cos(2*M_PI*x(0))*sin(2*M_PI*x(1))/2; 
		auto dpsi_dy = sin(M_PI*x(0))*cos(M_PI*x(1))/4 + quadratic*sin(2*M_PI*x(0))*cos(2*M_PI*x(1))/2; 
		return Omega(0)*dpsi_dx + Omega(1)*dpsi_dy 
			+ total_val*(sin(M_PI*x(0))*sin(M_PI*x(1)) + quadratic*sin(2*M_PI*x(0))*sin(2*M_PI*x(1)) + 2.0)/4/M_PI
			- scattering_val*(sin(M_PI*x(0))*sin(M_PI*x(1)) + 2./3*sin(2*M_PI*x(0))*sin(2*M_PI*x(1)) + 2.0)/4/M_PI; 
	};
	auto inflow_mms = [](const mfem::Vector &x, const mfem::Vector &Omega) {
		return 2.0/4/M_PI; 
	};
	FormTransportSource(fes, quad, psi_ext, qmms, inflow_mms, source);

	mfem::ParBilinearForm Ms_form(&fes); 
	Ms_form.AddDomainIntegrator(new mfem::MassIntegrator(scattering)); 
	Ms_form.Assemble(); 
	Ms_form.Finalize(); 

	InverseAdvectionOperator Linv(fes, quad, psi_ext, total, inflow); 

	mfem::ParGridFunction phi(&fes), phi_old(&fes); 
	phi_old = 0.0; 
	for (auto it=0; it<50; it++) {		
		Ms_form.Mult(phi_old, phi); 
		D.MultTranspose(phi, psi); 
		psi += source; 
		Linv.Mult(psi, psi); 
		D.Mult(psi, phi); 

		phi_old -= phi; 
		double norm = sqrt(InnerProduct(MPI_COMM_WORLD, phi_old, phi_old)); 
		phi_old = phi; 
		if (norm < 1e-10) break; 
	}

	auto exsol = [](const mfem::Vector &x) {
		return sin(M_PI*x(0))*sin(M_PI*x(1)) + 2./3*sin(2*M_PI*x(0))*sin(2*M_PI*x(1)) + 2.0; 
	};
	mfem::FunctionCoefficient exact_c(exsol); 
	return phi.ComputeL2Error(exact_c);
}

TEST(MMS, LinearTransport2Dp1) {
	auto Ne = 10; 
	const auto fe_order = 1; 
	mfem::Mesh mesh1 = mfem::Mesh::MakeCartesian2D(Ne,Ne, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	mfem::Mesh mesh2 = mfem::Mesh::MakeCartesian2D(2*Ne, 2*Ne, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	double E1 = LinearTransportError(mesh1, fe_order); 
	double E2 = LinearTransportError(mesh2, fe_order); 
	double ooa = log2(E1/E2); 
	bool within_bounds = (fe_order+1 - ooa) < .2; 
	EXPECT_TRUE(within_bounds); 
}

TEST(MMS, LinearTransport2DTRI) {
	auto Ne = 10; 
	const auto fe_order = 1; 
	mfem::Mesh mesh1 = mfem::Mesh::MakeCartesian2D(Ne,Ne, mfem::Element::TRIANGLE, true, 1.0, 1.0, false); 
	mfem::Mesh mesh2 = mfem::Mesh::MakeCartesian2D(2*Ne, 2*Ne, mfem::Element::TRIANGLE, true, 1.0, 1.0, false); 
	double E1 = LinearTransportError(mesh1, fe_order); 
	double E2 = LinearTransportError(mesh2, fe_order); 
	double ooa = log2(E1/E2); 
	bool within_bounds = (fe_order+1 - ooa) < .2; 
	EXPECT_TRUE(within_bounds); 
}

TEST(MMS, LinearTransport2Dp2) {
	auto Ne = 10; 
	const auto fe_order = 2; 
	mfem::Mesh mesh1 = mfem::Mesh::MakeCartesian2D(Ne,Ne, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	mfem::Mesh mesh2 = mfem::Mesh::MakeCartesian2D(2*Ne, 2*Ne, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	double E1 = LinearTransportError(mesh1, fe_order); 
	double E2 = LinearTransportError(mesh2, fe_order); 
	double ooa = log2(E1/E2); 
	bool within_bounds = (fe_order+1 - ooa) < .2; 
	EXPECT_TRUE(within_bounds); 
}

TEST(MMS, LinearTransport2Dp3) {
	auto Ne = 10; 
	const auto fe_order = 3; 
	mfem::Mesh mesh1 = mfem::Mesh::MakeCartesian2D(Ne,Ne, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	mfem::Mesh mesh2 = mfem::Mesh::MakeCartesian2D(2*Ne, 2*Ne, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	double E1 = LinearTransportError(mesh1, fe_order); 
	double E2 = LinearTransportError(mesh2, fe_order); 
	double ooa = log2(E1/E2); 
	bool within_bounds = (fe_order+1 - ooa) < .2; 
	EXPECT_TRUE(within_bounds); 
}
