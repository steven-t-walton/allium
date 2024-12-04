#include "sweep.hpp"
#include "multigroup.hpp"
#include "fixed_point.hpp"
#include "yaml-cpp/yaml.h"
#include "comment_stream.hpp"
#include "io.hpp"
#include <random>

int main(int argc, char *argv[]) {
	mfem::Mpi::Init();
	const auto rank = mfem::Mpi::WorldRank();
	const auto root = mfem::Mpi::Root();
	const auto num_proc = mfem::Mpi::WorldSize();

	// stream for output to terminal
	mfem::OutStream par_out(std::cout);
	// enable only for root so non-root procs don't clutter cout 
	if (rank!=0) par_out.Disable();

	// make mfem print everything with 
	// yaml comment preceeding 
	// helps keep output yaml parse-able 
	CommentStreamBuf comm_buf(mfem::out, '#'); 

	// parse cmdline arguments 
	mfem::OptionsParser args(argc, argv); 
	mfem::Array<int> part;
	double opacity = 1.0;
	int sn_order = 1;
	args.AddOption(&part, "-p", "--part", "parallel owned elements", true);
	args.AddOption(&opacity, "-o", "--opacity", "opacity");
	args.AddOption(&sn_order, "-s", "--sn_order", "number of angles");
	args.Parse(); 
	if (!args.Good()) {
		args.PrintUsage(par_out); 
		return 1; 
	}
	if (root) {
		args.PrintOptions(mfem::out); 
		mfem::out << std::endl; 		
	}

	// YAML output 
	YAML::Emitter out(par_out);
	out.SetDoublePrecision(8); 
	out << YAML::BeginMap; 

	const auto Ne = part.Size();
	auto smesh = mfem::Mesh::MakeCartesian1D(Ne, 1.0);

	mfem::ParMesh mesh(MPI_COMM_WORLD, smesh, part.GetData());
	const auto dim = mesh.Dimension();
	mfem::L2_FECollection fec(1, dim, mfem::BasisType::GaussLobatto);
	mfem::L2_FECollection fec0(0, dim);
	mfem::ParFiniteElementSpace fes(&mesh, &fec); 
	mfem::ParFiniteElementSpace fes0(&mesh, &fec0);
	LevelSymmetricQuadrature lvlquad(4, dim);
	SingleAngleQuadratureRule quad(lvlquad, 0);
	const auto grid = MultiGroupEnergyGrid::MakeGray(0.0, 1.0);

	TransportVectorExtents psi_ext(grid.Size(), quad.Size(), fes.GetVSize());
	const auto psi_size = TotalExtent(psi_ext);
	mfem::Vector psi(psi_size), psi_old(psi_size), source(psi_size);
	psi = 0.0;

	ConstantGrayMGCoefficient total(opacity, 1);

	BoundaryConditionMap bc_map;
	const auto &bdr_attr = mesh.bdr_attributes;
	for (const auto &attr : bdr_attr) {
		bc_map[attr] = INFLOW;
	}

	InverseAdvectionOperator Linv(fes, quad, total, bc_map, 7);
	Linv.UseParallelBlockJacobi(true);
	Linv.WriteGlobalGraphToDot("graph");
	Linv.Exchange(psi);

	ConstantPhaseSpaceCoefficient inflow_coef(1.0);
	ConstantPhaseSpaceCoefficient source_coef(0.0);
	FormTransportSource(fes, quad, grid, source_coef, inflow_coef, source);

	int it;
	double norm;
	mfem::StopWatch timer; 
	timer.Start();
	for (it=1; it<200; it++) {
		psi_old = psi;
		Linv.Mult(source, psi);
		psi_old -= psi; 
		norm = mfem::GlobalLpNorm(2.0, psi_old.Norml2(), MPI_COMM_WORLD);
		if (norm < 1e-10) break;
	}
	timer.Stop();
	out << YAML::Key << "Ne" << YAML::Value << smesh.GetNE();
	out << YAML::Key << "it" << YAML::Value << it; 
	out << YAML::Key << "norm" << YAML::Value << norm; 
	out << YAML::Key << "time" << YAML::Value << io::FormatScientific(timer.RealTime());
	out << YAML::EndMap << YAML::Newline; 

// 	mfem::VisItDataCollection dc("solution", &mesh);
// 	mfem::ParGridFunction gf(&fes, psi, 0);
// 	dc.RegisterField("psi", &gf);
// 	dc.Save();
}