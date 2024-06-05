#include "gtest/gtest.h"
#include "linalg.hpp"
#include "transport_op.hpp"
#include "sweep.hpp"

TEST(Linalg, BlockInverse) {
	mfem::Mesh smesh = mfem::Mesh::MakeCartesian2D(10,10,mfem::Element::QUADRILATERAL); 
	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(1, dim, mfem::BasisType::GaussLegendre); 
	mfem::ParFiniteElementSpace fes(&mesh, &fec); 
	mfem::ParBilinearForm Mform(&fes); 
	Mform.AddDomainIntegrator(new mfem::MassIntegrator); 
	Mform.Assemble(); 
	Mform.Finalize(); 
	auto M = std::unique_ptr<mfem::HypreParMatrix>(Mform.ParallelAssemble());
	auto iM = std::unique_ptr<mfem::HypreParMatrix>(ElementByElementBlockInverse(fes, *M));  

	// ensure product is identity matrix 
	auto eye = std::unique_ptr<mfem::HypreParMatrix>(mfem::ParMult(iM.get(), M.get(), true));
	const auto &I = eye->GetDiagMemoryI(); 
	const auto &J = eye->GetDiagMemoryJ(); 
	const auto &data = eye->GetDiagMemoryData(); 
	for (auto i=0; i<eye->GetNumRows(); i++) {
		for (auto j=I[i]; j<I[i+1]; j++) {
			auto col = J[j]; 
			EXPECT_EQ(i, col); 
			EXPECT_DOUBLE_EQ(data[j], 1.0); 
		}
	}
}

TEST(Linal, BlockLDUInverse) {
	mfem::Mesh smesh = mfem::Mesh::MakeCartesian2D(20,20, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh); 
	const auto dim = mesh.Dimension();
	LevelSymmetricQuadrature quad(6, dim); 

	const int fe_order = 1; 
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
	const auto phi_size = TotalExtent(phi_ext); 
	mfem::Array<int> offsets(3); 
	offsets[0] = 0; 
	offsets[1] = phi_size; 
	offsets[2] = psi_size; 	
	offsets.PartialSum(); 
	mfem::BlockVector b(offsets), x(offsets); 
	b = 0.0; 

	DiscreteToMoment D(quad, psi_ext, phi_ext); 
	mfem::ScaledOperator Dneg(&D, -1.0); 
	mfem::TransposeOperator DT(D); 
	mfem::IdentityOperator Iphi(phi_size); 

	TransportVectorView source_view(b.GetBlock(1).GetData(), psi_ext); 

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
	FunctionGrayCoefficient qmms_coef(qmms); 
	FunctionGrayCoefficient inflow_coef(inflow_mms); 
	mfem::Array<double> energy_grid(2); 
	FormTransportSource(fes, quad, energy_grid, qmms_coef, inflow_coef, source_view);

	mfem::ParBilinearForm Ms_form(&fes); 
	Ms_form.AddDomainIntegrator(new mfem::MassIntegrator(scattering)); 
	Ms_form.Assemble(); 
	Ms_form.Finalize(); 

	mfem::GridFunction total_data(&fes); 
	total_data.ProjectCoefficient(total); 
	BoundaryConditionMap bc_map;
	const auto &bdr_attr = mesh.bdr_attributes;
	for (const auto &attr : bdr_attr) {
		bc_map[attr] = INFLOW;
	}
	InverseAdvectionOperator Linv(fes, quad, total_data, bc_map); 

	TransportOperator T(D, Linv, Ms_form, x.GetBlock(1)); 
	mfem::GMRESSolver Sinv; 
	Sinv.SetOperator(T); 
	Sinv.SetAbsTol(1e-10); 
	Sinv.SetMaxIter(100); 
	BlockLDUInverseOperator ldu_inv(Sinv, Linv, Dneg, DT); 
	ldu_inv.Mult(b, x); 

	mfem::ParGridFunction phi(&fes, x.GetBlock(0), 0); 

	auto exsol = [](const mfem::Vector &x) {
		return sin(M_PI*x(0))*sin(M_PI*x(1)) + 2./3*sin(2*M_PI*x(0))*sin(2*M_PI*x(1)) + 2.0; 
	};
	mfem::FunctionCoefficient exact_c(exsol); 
	double err = phi.ComputeL2Error(exact_c); 
	EXPECT_LT(err, 0.01); 
}