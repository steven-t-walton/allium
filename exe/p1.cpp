#include "mfem.hpp"
#include "tvector.hpp"
#include "angular_quadrature.hpp"
#include "sweep.hpp"
#include "p1diffusion.hpp"
#include "linalg.hpp"

using HypreParMatrixPtr = std::unique_ptr<mfem::HypreParMatrix>; 

// wrap a HypreParMatrix to make it look like a solver 
// used to precondition mass matrix inverse with an approximate inverse mass matrix 
class MockSolver : public mfem::Solver
{
private:
	mfem::HypreParMatrix &inv; 
public:
	MockSolver(mfem::HypreParMatrix &A) : inv(A), mfem::Solver(A.Height(), false) { } 
	void Mult(const mfem::Vector &x, mfem::Vector &y) const { inv.Mult(x,y); }
	void SetOperator(const mfem::Operator&) { } // no op 
};

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

	// build source term
	// even this simple problem cannot be scalably solved 
	// with approximate Schur complement-based block preconditioners 
	mfem::LinearForm fform(&fes); 
	mfem::ConstantCoefficient source_coef(1.0); 
	fform.AddDomainIntegrator(new mfem::DomainLFIntegrator(source_coef)); 
	fform.Assemble(); 

	// approximate value of \int |Omega.n| dOmega / 4pi 
	// typically computed approximately with SN quadrature 
	// exact value of 0.5 is ok for this 
	double alpha = 0.5; 

	// forms block system for [J, phi] 
	// [ 3 Mt + F1  -(D + F2)^T ] [J]    = [ 3 g ]
	// [   D + F2      Ma + P   ] [phi]  = [  f  ] 
	// where F1, F2, and P couple neighboring elements 
	// applies "robin" boundary conditions (zero inflow BCs)
	// see src/p1diffusion.cpp
	mfem::ParBilinearForm Mtform(&vfes);
	mfem::ProductCoefficient total3(3.0, total); 
	Mtform.AddDomainIntegrator(new mfem::VectorMassIntegrator(total3));
	Mtform.AddInteriorFaceIntegrator(new DGVectorJumpJumpIntegrator(1.0/2/alpha)); 
	Mtform.AddBdrFaceIntegrator(new DGVectorJumpJumpIntegrator(1.0/2/alpha)); 
	Mtform.Assemble(); 
	Mtform.Finalize();  
	mfem::HypreParMatrix *Mt = Mtform.ParallelAssemble(); 

	mfem::ParBilinearForm Maform(&fes); 
	mfem::ConstantCoefficient alpha_c(alpha/2); 
	Maform.AddDomainIntegrator(new mfem::MassIntegrator(absorption)); 
	Maform.AddInteriorFaceIntegrator(new PenaltyIntegrator(alpha/2, false)); 
	Maform.AddBdrFaceIntegrator(new mfem::BoundaryMassIntegrator(alpha_c)); 
	Maform.Assemble(); 
	Maform.Finalize(); 
	mfem::HypreParMatrix *Ma = Maform.ParallelAssemble();

	mfem::ParMixedBilinearForm Dform(&vfes, &fes); 
	mfem::ConstantCoefficient neg_one(-1.0); 
	Dform.AddDomainIntegrator(new mfem::TransposeIntegrator(new mfem::GradientIntegrator(neg_one))); 
	Dform.AddInteriorFaceIntegrator(new DGJumpAverageIntegrator); 
	Dform.AddBdrFaceIntegrator(new DGJumpAverageIntegrator(0.5)); 
	Dform.Assemble(); 
	Dform.Finalize(); 
	mfem::HypreParMatrix *D = Dform.ParallelAssemble(); 
	mfem::HypreParMatrix *DT = D->Transpose(); 
	(*DT) *= -1.0; 

	auto *p1_op = new mfem::BlockOperator(offsets); 
	p1_op->SetBlock(0,0, Mt); 
	p1_op->SetBlock(0,1, DT); 
	p1_op->SetBlock(1,0, D); 
	p1_op->SetBlock(1,1, Ma); 
	p1_op->owns_blocks = 1; 

	// --- form approximate schur complement with row sum of Mt --- 
	// form approximate inverse of Mt via inverse of row sum 
	// poor preconditioner for conjugate gradient applied to Mt 
	// amg(S) not great either... 
	// mfem::SparseMatrix diag; 
	// Mt->GetDiag(diag); // get processor local CSR 
	// mfem::Vector row_sums(diag.Width()); 
	// diag.GetRowSums(row_sums); 
	// row_sums.Reciprocal(); // element-wise 1/row_sums 
	// mfem::SparseMatrix iMt_local(row_sums); 
	// auto iMt = HypreParMatrixPtr(
	// 	std::make_unique<mfem::HypreParMatrix>(vfes.GetComm(), vfes.GlobalVSize(), vfes.GetDofOffsets(), &iMt_local)
	// ); 

	// --- second type of approximate schur complement --- 
	// invert mass matrix without face interior terms 
	mfem::ParBilinearForm iMtform(&vfes);
	iMtform.AddDomainIntegrator(new mfem::VectorMassIntegrator(total3));
	// Mtform.AddInteriorFaceIntegrator(new DGVectorJumpJumpIntegrator(1.0/2/alpha)); // <-- decouple by removing this term 
	iMtform.AddBdrFaceIntegrator(new DGVectorJumpJumpIntegrator(1.0/2/alpha)); 
	iMtform.Assemble(); 
	iMtform.Finalize();  
	auto Mt_diag = HypreParMatrixPtr(iMtform.ParallelAssemble()); 
	auto iMt = HypreParMatrixPtr(ElementByElementBlockInverse(vfes, *Mt_diag)); 

	// matmult for Schur complement 
	auto iMtDT = HypreParMatrixPtr(mfem::ParMult(iMt.get(), DT, true)); // true = copy row/col starts vector
	auto DiMtDT = HypreParMatrixPtr(mfem::ParMult(D, iMtDT.get(), true)); 
	auto S = HypreParMatrixPtr(mfem::ParAdd(DiMtDT.get(), Ma)); 

	// apply AMG to Schur complement 
	mfem::HypreBoomerAMG amg(*S); 
	amg.SetPrintLevel(0); 

	// krylov for Mt block 
	mfem::CGSolver cg(MPI_COMM_WORLD); 
	cg.SetAbsTol(1e-12); 
	cg.SetMaxIter(100); 
	cg.SetPrintLevel(1); 
	cg.SetOperator(*Mt); 
	MockSolver iMt_solver(*iMt); 
	cg.SetPreconditioner(iMt_solver); // precondition with approximate mass inverse 

	// lower block triangular preconditioner 
	mfem::BlockLowerTriangularPreconditioner prec(offsets); 
	prec.SetBlock(0,0, &cg); // solve with preconditioned conjugate gradient 
	// prec.SetBlock(0,0, iMt.get()); // directly use iMt approximation 
	prec.SetBlock(1,1, &amg); 
	prec.SetBlock(1,0, D); 

	// flexible GMRES since Krylov used for Mt 
	mfem::FGMRESSolver solver(MPI_COMM_WORLD); 
	solver.SetAbsTol(1e-10); 
	solver.SetMaxIter(500); 
	solver.SetPrintLevel(1); 
	solver.SetOperator(*p1_op); 
	solver.SetPreconditioner(prec); 

	// create block source and solution vectors 
	mfem::BlockVector b(offsets), x(offsets); 
	x = 0.0; 
	b.GetBlock(0) = 0.0; 
	b.GetBlock(1) = fform; 

	// solve 
	solver.Mult(b, x); 
	printf("it = %d, norm = %.3e\n", solver.GetNumIterations(), solver.GetFinalNorm()); 

	// --- uncomment for superlu solve --- 
	// uses mfem::Hypre::HypreParMatrixFromBlocks to form monolithic system 
	// auto mono = HypreParMatrixPtr(BlockOperatorToMonolithic(*p1_op)); 

	// // solve with superlu 
	// mfem::SuperLURowLocMatrix mono_slu(*mono); 
	// mfem::SuperLUSolver slu(mono_slu); 
	// slu.SetPrintStatistics(false); 
	// slu.Mult(b, x); 

	// extract solution components 
	mfem::ParGridFunction u(&fes, x.GetBlock(1)); 
	mfem::ParGridFunction v(&vfes, x.GetBlock(0)); 

	// plot with paraview 
	mfem::ParaViewDataCollection dc("solution", &mesh); 
	dc.RegisterField("u", &u); 
	dc.RegisterField("v", &v); 
	dc.Save(); 
}