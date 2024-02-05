#include "mfem.hpp"
#include "tvector.hpp"
#include "angular_quadrature.hpp"
#include "sweep.hpp"
#include "p1diffusion.hpp"

using HypreParMatrixPtr = std::unique_ptr<mfem::HypreParMatrix>; 

int main(int argc, char *argv[]) {
	mfem::Mpi::Init(argc, argv); 
	mfem::Hypre::Init(); 

	const auto rank = mfem::Mpi::WorldRank(); 
	const auto num_proc = mfem::Mpi::WorldSize(); 
	const bool root = rank == 0; 

	// out will print to terminal only for root processor 
	mfem::OutStream out(std::cout); 
	if (!root) out.Disable(); 

	auto Ne = 10;
	auto fe_order = 1; 
	mfem::OptionsParser args(argc, argv); 
	args.AddOption(&Ne, "-n", "--Ne", "number of elements in each dimension"); 
	args.AddOption(&fe_order, "-p", "--fe_order", "finite element order"); 
	args.Parse(); 
	if (!args.Good()) {
		args.PrintUsage(out); 
		return 1; 
	}
	args.PrintOptions(out); 

	mfem::Mesh smesh = mfem::Mesh::MakeCartesian2D(Ne, Ne, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh); 
	const auto dim = mesh.Dimension(); 

	// build DG scalar and vector space 
	mfem::L2_FECollection fec(fe_order, dim, mfem::BasisType::GaussLegendre); 
	mfem::ParFiniteElementSpace fes(&mesh, &fec); 
	mfem::ParFiniteElementSpace vfes(&mesh, &fec, dim); 

	// block operator offsets [ J, phi ] 
	mfem::Array<int> offsets(3); 
	offsets[0] = 0; 
	offsets[1] = vfes.GetVSize(); 
	offsets[2] = fes.GetVSize(); 
	offsets.PartialSum(); 

	mfem::ConstantCoefficient total(1.0); 
	mfem::ConstantCoefficient absorption(0.5); 

	// forms block system for [J, phi] 
	// [ 3 Mt + F1  -(D + F2)^T ] [J]    = [ 3 g ]
	// [   D + F2      Ma + P   ] [phi]  = [  f  ] 
	// where F1, F2, and P couple neighboring elements 
	// applies "robin" boundary conditions (zero inflow BCs)
	// see src/p1diffusion.cpp
	auto p1_op = std::unique_ptr<mfem::BlockOperator>(CreateP1DiffusionDiscretization(fes, vfes, total, absorption)); 
	// alias for unique ptr to HypreParMatrix (auto deletes at end) 
	// uses mfem::Hypre::HypreParMatrixFromBlocks to form monolithic system 
	auto mono = HypreParMatrixPtr(BlockOperatorToMonolithic(*p1_op)); 

	// build source term
	// even this simple problem cannot be scalably solved 
	// with approximate Schur complement-based block preconditioners 
	mfem::LinearForm fform(&fes); 
	mfem::ConstantCoefficient source_coef(1.0); 
	fform.AddDomainIntegrator(new mfem::DomainLFIntegrator(source_coef)); 
	fform.Assemble(); 

	mfem::BlockVector b(offsets), x(offsets); 
	b.GetBlock(0) = 0.0; 
	b.GetBlock(1) = fform; 

	// solve with superlu 
	mfem::SuperLURowLocMatrix mono_slu(*mono); 
	mfem::SuperLUSolver solver(mono_slu); 
	solver.SetPrintStatistics(false); 
	solver.Mult(b, x); 

	// extract solution components 
	mfem::ParGridFunction u(&fes, x.GetBlock(1)); 
	mfem::ParGridFunction v(&vfes, x.GetBlock(0)); 

	// plot with paraview 
	mfem::ParaViewDataCollection dc("solution", &mesh); 
	dc.RegisterField("u", &u); 
	dc.RegisterField("v", &v); 
	dc.Save(); 
}