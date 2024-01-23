#include "gtest/gtest.h"
#include "linalg.hpp"

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