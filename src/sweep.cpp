#include "sweep.hpp"
#include "config.hpp"
#include "log.hpp"
#include "lumping.hpp"
#include "multigroup.hpp"
#include <cstring>
#include <omp.h>

DenseSnKernel::DenseSnKernel(mfem::ParFiniteElementSpace &fes, const AngularQuadrature &quad,
	const DGLocalMatrices &mats, const SweepScheduler &sched)
	: fes(fes), quad(quad), mats(mats), sched(sched),
	  psi_ext(mats.NumGroups(), quad.Size(), fes.GetVSize()),
	  scratch(omp_get_max_threads())
{ }

void DenseSnKernel::Bind(const mfem::Vector &_source, mfem::Vector &_psi, mfem::Vector &_psi_fnbr)
{
	source = _source.GetData();
	psi = _psi.GetData();
	psi_fnbr = _psi_fnbr.GetData();
	// thread count may change between sweeps
	if (scratch.size() < omp_get_max_threads()) scratch.resize(omp_get_max_threads());
}

void DenseSnKernel::LogFixup()
{
	int success = 0, failed = 0;
	for (auto &s : scratch) {
		success += s.fixup_success; failed += s.fixup_failed;
		s.fixup_success = s.fixup_failed = 0;
	}
	EventLog.Log("fixup applied", success);
	EventLog.Log("fixup failed", failed);
}

void DenseSnKernel::Solve(int e, int set, std::span<const SweepEdge> in)
{
	auto &[grad, A, lu, rhs, sol, psi2, nor, dofs, dofs2, fixup_success, fixup_failed] = scratch[omp_get_thread_num()];
	const auto dim = fes.GetMesh()->Dimension();
	const auto Ne = fes.GetNE();
	const auto G = psi_ext.extent(0);
	TransportVectorView psi_view(psi, psi_ext);
	ConstTransportVectorView source_view(source, psi_ext);

	fes.GetElementDofs(e, dofs);
	rhs.SetSize(dofs.Size());
	for (const auto a : sched.AngleSet(set)) {
		const auto &Omega = quad.GetOmega(a);

		// --- assemble gradient term ---
		grad.SetSize(dofs.Size());
		grad = 0.0;
		for (auto d=0; d<dim; d++) {
			grad.Add(Omega(d), mats.Grad(d,e));
		}

		// --- assemble outflow face terms ---
		for (const auto f : sched.ElementFaces(e)) {
			const auto idx = sched.ElementSide(e, f);
			sched.OutwardNormal(f, idx, nor);
			const double dot = Omega * nor;
			// outflow
			if (dot > 0) {
				grad.Add(dot, mats.Face(f, idx, idx));
			}
		}

		// --- assemble and solve all groups ---
		for (int g=0; g<G; g++) {
			// copy group-independent terms
			A = grad;
			// add total mass matrix for group g
			A.Add(1.0, mats.Mass(g,e));

			// --- load source into local vector ---
			for (auto i=0; i<dofs.Size(); i++) { rhs[i] = source_view(g,a,dofs[i]); }

			// add inflow terms to RHS
			for (const auto &edge : in) {
				const auto idx = edge.side;
				const auto nbr_idx = (idx + 1) % 2;
				sched.OutwardNormal(edge.face, idx, nor);
				const double dot = Omega * nor;

				if (edge.kind == SweepEdge::REFLECT) {
					const auto nbr_a = quad.GetReflectedAngleIndex(a, nor);
					psi2.SetSize(dofs.Size());
					for (auto i=0; i<dofs.Size(); i++) { psi2(i) = psi_view(g, nbr_a, dofs[i]); }
					mats.Face(edge.face, idx, idx).AddMult(psi2, rhs, -dot);
				}

				else { // inflow from neighbor
					// face neighbor data
					if (edge.kind == SweepEdge::GHOST) {
						fes.GetFaceNbrElementVDofs(edge.elem, dofs2);
						psi2.SetSize(dofs2.Size());
						for (int i=0; i<dofs2.Size(); i++) { psi2(i) = psi_fnbr[GhostIndex(g, a, dofs2[i])]; }
					}
					// local data
					else {
						fes.GetElementDofs(edge.elem, dofs2);
						psi2.SetSize(dofs2.Size());
						for (int i=0; i<dofs2.Size(); i++) { psi2(i) = psi_view(g, a, dofs2[i]); }
					}
					mats.Face(edge.face, idx, nbr_idx).AddMult(psi2, rhs, -dot);
				}
			}

			if (time_dependent) {
				A.Add(time_absorption, mats.TimeMass(e));
			}
			// solve, solution overwritten into rhs, matrix is modified
			lu = A;
			sol = rhs;
			mfem::LinearSolve(lu, sol.GetData());

			if (fixup_op) {
				const auto applied = fixup_op->Apply(A, rhs, sol);
				if (applied==1) fixup_success++;
				else if (applied == -1) fixup_failed++;
				if (fixup_monitor and applied) (*fixup_monitor)(e + g*Ne)++;
			}

			// scatter back
			for (int i=0; i<dofs.Size(); i++) { psi_view(g, a, dofs[i]) = sol(i); }
		}
	}
}

std::size_t DenseSnKernel::PackSize(int e, int set) const
{
	return sizeof(double) * sched.AngleSet(set).size() * psi_ext.extent(0) * fes.GetFE(e)->GetDof();
}

void DenseSnKernel::Pack(int e, int set, char *buf) const
{
	ConstTransportVectorView psi_view(psi, psi_ext);
	mfem::Array<int> dofs;
	fes.GetElementDofs(e, dofs);
	for (const auto a : sched.AngleSet(set)) {
		for (int g=0; g<psi_ext.extent(0); g++) {
			for (int i=0; i<dofs.Size(); i++) {
				std::memcpy(buf, &psi_view(g,a,dofs[i]), sizeof(double));
				buf += sizeof(double);
			}
		}
	}
}

void DenseSnKernel::Unpack(int ghost_e, int set, const char *buf)
{
	mfem::Array<int> dofs;
	fes.GetFaceNbrElementVDofs(ghost_e, dofs);
	for (const auto a : sched.AngleSet(set)) {
		for (int g=0; g<psi_ext.extent(0); g++) {
			for (int i=0; i<dofs.Size(); i++) {
				std::memcpy(&psi_fnbr[GhostIndex(g, a, dofs[i])], buf, sizeof(double));
				buf += sizeof(double);
			}
		}
	}
}

InverseAdvectionOperator::InverseAdvectionOperator(mfem::ParFiniteElementSpace &_fes, const AngularQuadrature &_quad,
	MultiGroupCoefficient &total, const BoundaryConditionMap &bc_map, int use_lumping)
	: fes(_fes), mesh(*_fes.GetParMesh()), quad(_quad), total(total), lump(use_lumping),
	  mats(_fes, total, use_lumping),
	  scheduler(*_fes.GetParMesh(), _quad, bc_map),
	  kernel(_fes, _quad, mats, scheduler)
{
	if (lump) {
		const auto *fec = fes.FEColl();
		const auto btype = dynamic_cast<const mfem::L2_FECollection*>(fec)->GetBasisType();
		if (btype != mfem::BasisType::GaussLobatto) {
			MFEM_ABORT("only lobatto supported with lumping");
		}
	}
	const auto Ndof = fes.GetVSize(); // local degrees of freedom
	const auto Nomega = quad.Size(); // number of angles
	const auto G = total.GetVDim();
	psi_ext = TransportVectorExtents(G, Nomega, Ndof); // multi index based on energy, angle, space
	// set operator sizes to size of psi
	height = width = TotalExtent(psi_ext);

	// allocate parallel exchange buffer
	const auto Ndof_fnbr = fes.GetFaceNbrVSize(); // spatial DOF in ghost buffer
	psi_fnbr_ext = TransportVectorExtents(G, Nomega, Ndof_fnbr);
	psi_fnbr.SetSize(TotalExtent(psi_fnbr_ext));
	psi_fnbr = 0.0;
}

void InverseAdvectionOperator::Mult(const mfem::Vector &source, mfem::Vector &psi) const
{
	if (pbj) {
		// if more than one sweep, have to save the source term
		// to be correct for in place calls to Mult
		if (num_pbj_sweeps > 1) {
			mfem::Vector tmp(source);
			for (int i=0; i<num_pbj_sweeps; i++) {
				Mult_BlockJacobi(tmp, psi);
			}
		} else {
			// do 1 PBJ sweep in place
			Mult_BlockJacobi(source, psi);
		}
	}
	else Mult_Upwind(source, psi);
}

void InverseAdvectionOperator::SetupKernel(const mfem::Vector &source, mfem::Vector &psi) const
{
	assert(source.Size() == Width());
	assert(psi.Size() == Height());
	kernel.Bind(source, psi, psi_fnbr);
	kernel.time_dependent = is_time_dependent;
	kernel.time_absorption = time_absorption;
	kernel.fixup_op = (apply_fixup) ? fixup_op : nullptr;
	kernel.fixup_monitor = fixup_monitor;
}

void InverseAdvectionOperator::Mult_Upwind(const mfem::Vector &source, mfem::Vector &psi) const
{
	mfem::StopWatch timer;
	timer.Start();
	SetupKernel(source, psi);
	scheduler.Sweep(kernel);
	kernel.LogFixup();
	timer.Stop();
	TimingLog.Log("sweep", timer.RealTime());

	if (mesh.GetMyRank()==0) EventLog.Register("sweeps");
}

void InverseAdvectionOperator::Mult_BlockJacobi(const mfem::Vector &source, mfem::Vector &psi) const
{
	mfem::StopWatch timer;
	timer.Start();
	SetupKernel(source, psi);
	scheduler.Sweep(kernel, true);
	kernel.LogFixup();
	ExchangeGhost(psi);
	timer.Stop();
	TimingLog.Log("pbj sweep", timer.RealTime());

	if (mesh.GetMyRank()==0) EventLog.Register("pbj sweeps");
}

void InverseAdvectionOperator::ExchangeGhost(const mfem::Vector &psi) const
{
	const auto Nomega = quad.Size();
	const auto G = psi_ext.extent(0);

	ConstTransportVectorView psi_view(psi.GetData(), psi_ext);
	if (fes.GetFaceNbrVSize() > 0) {
		int *send_offset = fes.send_face_nbr_ldof.GetI();
		const int *d_send_offset = fes.send_face_nbr_ldof.GetJ();
		int *recv_offset = fes.face_nbr_ldof.GetI();
		MPI_Comm comm = fes.GetComm();
		const int num_face_nbrs = mesh.GetNFaceNeighbors();

		// load data into send buffer
		const auto num_send_dofs = fes.send_face_nbr_ldof.Size_of_connections();
		mfem::Vector send_buffer(num_send_dofs * G * Nomega);
		for (int i=0; i<num_send_dofs; i++) {
			for (int g=0; g<G; g++) {
				for (int a=0; a<Nomega; a++) {
					send_buffer(a + g*Nomega + Nomega*G*i) = psi_view(g,a,d_send_offset[i]);
				}
			}
		}

		auto *requests = new MPI_Request[2*num_face_nbrs];
		auto *send_requests = requests;
		auto *recv_requests = requests + num_face_nbrs;
		auto *statuses = new MPI_Status[num_face_nbrs];
		for (int fn=0; fn<num_face_nbrs; fn++) {
			const int nbr_rank = mesh.GetFaceNbrRank(fn);
			int tag = 0;
			MPI_Isend(&send_buffer.GetData()[send_offset[fn]*G*Nomega],
				(send_offset[fn+1] - send_offset[fn])*G*Nomega, MPI_DOUBLE,
				nbr_rank, tag, comm, &send_requests[fn]);
			MPI_Irecv(&psi_fnbr.GetData()[recv_offset[fn]*G*Nomega],
				(recv_offset[fn+1] - recv_offset[fn])*G*Nomega,
				MPI_DOUBLE, nbr_rank, tag, comm, &recv_requests[fn]);
		}
		MPI_Waitall(num_face_nbrs, send_requests, statuses);
		MPI_Waitall(num_face_nbrs, recv_requests, statuses);
		delete[] requests; delete[] statuses;
	}
}

void InverseAdvectionOperator::SetTimeAbsorption(const double sigma)
{
	time_absorption = sigma;
	is_time_dependent = true;
	mats.AssembleTimeMass();
}

void InverseAdvectionOperator::UseFixup(bool use)
{
	apply_fixup = use;
	if (use and !fixup_op) MFEM_ABORT("no fixup operator set");
}

void InverseAdvectionOperator::WriteGlobalGraphToDot(std::string prefix) const
{
	const auto rank = mesh.GetMyRank();
	const auto G = psi_ext.extent(0);
	std::vector<std::string> omega_str = {"mu", "eta", "xi"};
	mfem::Vector opac(G);
	scheduler.WriteGlobalGraphToDot(prefix, [&](std::ostream &out, int e, int a) {
		const auto &Omega = quad.GetOmega(a);
		auto &trans = *mesh.GetElementTransformation(e);
		const auto &ip = mfem::Geometries.GetCenter(trans.GetGeometryType());
		total.Eval(opac, trans, ip);
		out << "rank=" << rank;
		out << ", opacity=" << opac(0);
		out << ", h=" << mesh.GetElementSize(e);
		for (int d=0; d<Omega.Size(); d++) {
			out << ", " << omega_str[d] << "=" << Omega(d);
		}
	});
}

void FormTransportSource(mfem::ParFiniteElementSpace &fes, AngularQuadrature &quad, 
	const mfem::Array<double> &energy_grid, PhaseSpaceCoefficient &source_coef, 
	PhaseSpaceCoefficient &inflow_coef, TransportVectorView source_view)
{
	const int G = energy_grid.Size() - 1; 
	for (int g=0; g<G; g++) {
		const auto e_low = energy_grid[g]; const auto e_high = energy_grid[g+1];
		const auto mid = (e_low + e_high)/2;
		source_coef.SetEnergy(e_low, e_high, mid); inflow_coef.SetEnergy(e_low, e_high, mid); 
		for (auto a=0; a<quad.Size(); a++) {
			const auto &Omega = quad.GetOmega(a); 
			source_coef.SetAngle(Omega); inflow_coef.SetAngle(Omega); 

			mfem::ParLinearForm bform(&fes); 
			bform.AddDomainIntegrator(new mfem::DomainLFIntegrator(source_coef)); 
			mfem::VectorConstantCoefficient Q(Omega); 
			bform.AddBdrFaceIntegrator(new mfem::BoundaryFlowIntegrator(inflow_coef, Q, -1.0, -0.5));
			bform.Assemble(); 
			for (int i=0; i<bform.Size(); i++) {
				source_view(g,a,i) = bform[i]; 
			}
		}		
	}
}

void FormTransportSource(mfem::FiniteElementSpace &fes, const AngularQuadrature &quad, 
	const MultiGroupEnergyGrid &energy_grid, PhaseSpaceCoefficient &source_coef, 
	PhaseSpaceCoefficient &inflow_coef, mfem::Vector &source)
{
	TransportVectorExtents psi_ext(energy_grid.Size(), quad.Size(), fes.GetVSize());
	auto source_view = TransportVectorView(source.GetData(), psi_ext);
	for (int g=0; g<energy_grid.Size(); g++) {
		source_coef.SetEnergy(energy_grid.LowerBound(g), energy_grid.UpperBound(g), 
			energy_grid.MeanEnergy(g));
		inflow_coef.SetEnergy(energy_grid.LowerBound(g), energy_grid.UpperBound(g), 
			energy_grid.MeanEnergy(g));		
		for (int a=0; a<quad.Size(); a++) {
			const auto &Omega = quad.GetOmega(a);
			source_coef.SetAngle(Omega); inflow_coef.SetAngle(Omega);

			mfem::LinearForm bform(&fes); 
			bform.AddDomainIntegrator(new mfem::DomainLFIntegrator(source_coef)); 
			mfem::VectorConstantCoefficient Q(Omega); 
			bform.AddBdrFaceIntegrator(new mfem::BoundaryFlowIntegrator(inflow_coef, Q, -1.0, -0.5));
			bform.Assemble(); 
			for (int i=0; i<bform.Size(); i++) {
				source_view(g,a,i) = bform[i]; 
			}
		}
	}
}

void AdvectionOperator::Mult(const mfem::Vector &x, mfem::Vector &y) const 
{
	auto &mesh = Linv.mesh;
	const auto dim = mesh.Dimension(); 
	auto &fes = Linv.fes; 
	const auto &psi_ext = Linv.psi_ext; 
	const auto &psi_fnbr_ext = Linv.psi_fnbr_ext; 
	const auto G = psi_ext.extent(0); 
	const auto &quad = Linv.quad; 
	const auto Nomega = quad.Size(); 
	const auto &sched = Linv.scheduler;
	const auto &mats = Linv.mats;

	TransportVectorView xview(x.GetData(), psi_ext); 
	TransportVectorView yview(y.GetData(), psi_ext); 
	y = 0.0; 

	auto &psi_fnbr = Linv.psi_fnbr; 
	psi_fnbr = 0.0; 
	if (fes.GetFaceNbrVSize() > 0) {
		int *send_offset = fes.send_face_nbr_ldof.GetI(); 
		const int *d_send_offset = fes.send_face_nbr_ldof.GetJ(); 
		int *recv_offset = fes.face_nbr_ldof.GetI(); 
		MPI_Comm comm = fes.GetComm(); 
		const int num_face_nbrs = mesh.GetNFaceNeighbors(); 

		// load data into send buffer 
		const auto num_send_dofs = fes.send_face_nbr_ldof.Size_of_connections(); 
		mfem::Vector send_buffer(num_send_dofs * G * Nomega); 
		for (int i=0; i<fes.send_face_nbr_ldof.Size_of_connections(); i++) {
			for (int g=0; g<G; g++) {
				for (int a=0; a<Nomega; a++) {
					send_buffer(a + g*Nomega + Nomega*G*i) = xview(g,a,d_send_offset[i]); 
				}
			}
		}

		auto *requests = new MPI_Request[2*num_face_nbrs]; 
		auto *send_requests = requests; 
		auto *recv_requests = requests + num_face_nbrs; 
		auto *statuses = new MPI_Status[num_face_nbrs]; 
		for (int fn=0; fn<num_face_nbrs; fn++) {
			const int nbr_rank = mesh.GetFaceNbrRank(fn); 
			int tag = 0; 
			MPI_Isend(&send_buffer.GetData()[send_offset[fn]*G*Nomega], 
				(send_offset[fn+1] - send_offset[fn])*G*Nomega, MPI_DOUBLE, 
				nbr_rank, tag, comm, &send_requests[fn]); 
			MPI_Irecv(&psi_fnbr.GetData()[recv_offset[fn]*G*Nomega], 
				(recv_offset[fn+1] - recv_offset[fn])*G*Nomega,
				MPI_DOUBLE, nbr_rank, tag, comm, &recv_requests[fn]); 
		}
		MPI_Waitall(num_face_nbrs, send_requests, statuses); 
		MPI_Waitall(num_face_nbrs, recv_requests, statuses); 
		delete[] requests; delete[] statuses; 
	}

	mfem::Array<int> vdofs, vdofs2; 
	mfem::Vector xlocal, xlocal2, ylocal, ylocal2, nor(dim); 
	mfem::DenseMatrix grad; 
	for (int g=0; g<G; g++) {
		for (int a=0; a<quad.Size(); a++) {
			const auto &Omega = quad.GetOmega(a); 
			for (int e=0; e<mesh.GetNE(); e++) {
				fes.GetElementDofs(e, vdofs); 
				xlocal.SetSize(vdofs.Size()); 
				ylocal.SetSize(vdofs.Size()); 
				for (int i=0; i<vdofs.Size(); i++) { xlocal(i) = xview(g,a,vdofs[i]); }
				mats.Mass(g,e).Mult(xlocal, ylocal); 
				if (Linv.is_time_dependent)
					mats.TimeMass(e).AddMult(xlocal, ylocal, Linv.time_absorption);
				grad.SetSize(vdofs.Size()); 
				grad = 0.0; 
				for (int d=0; d<dim; d++) {
					grad.Add(Omega(d), mats.Grad(d,e)); 
				}
				grad.AddMult(xlocal, ylocal); 
				for (int i=0; i<vdofs.Size(); i++) { yview(g,a,vdofs[i]) += ylocal(i); }
			}

			for (int f=0; f<mesh.GetNumFaces(); f++) {
				sched.OutwardNormal(f, 0, nor);
				const auto &info = mesh.GetFaceInformation(f); 
				mfem::FaceElementTransformations *trans; 
				if (info.IsShared()) {
					trans = mesh.GetSharedFaceTransformationsByLocalIndex(f, true); 
				} else {
					trans = mesh.GetFaceElementTransformations(f); 
				}
				const double dot = Omega * nor; 
				const auto e1 = trans->Elem1No; 
				fes.GetElementDofs(e1, vdofs); 
				xlocal.SetSize(vdofs.Size()); 
				ylocal.SetSize(vdofs.Size()); 
				for (int i=0; i<vdofs.Size(); i++) { xlocal(i) = xview(g,a,vdofs[i]); }

				ylocal = 0.0; 
				if (info.IsShared()) {
					const auto e2 = trans->Elem2No; 
					const auto idx = e2 - mesh.GetNE(); 
					fes.GetFaceNbrElementVDofs(idx, vdofs2); 
					xlocal2.SetSize(vdofs2.Size());  
					for (int i=0; i<vdofs2.Size(); i++) { xlocal2(i) = psi_fnbr[a + g*Nomega + Nomega*G*vdofs2[i]]; }
					if (dot >= 0) {
						mats.Face(f,0,0).AddMult(xlocal, ylocal, dot); 
					} else {
						mats.Face(f,0,1).AddMult(xlocal2, ylocal, dot); 
					}
				}

				else if (info.IsInterior()) {
					const auto e2 = trans->Elem2No; 
					fes.GetElementDofs(e2, vdofs2);
					xlocal2.SetSize(vdofs2.Size());  
					ylocal2.SetSize(vdofs2.Size()); 
					for (int i=0; i<vdofs2.Size(); i++) { xlocal2(i) = xview(g,a,vdofs2[i]); }
					ylocal2 = 0.0; 
					if (dot >= 0) { // Omega points 1 -> 2 
						// 1 gets outflow 
						mats.Face(f,0,0).AddMult(xlocal, ylocal, dot); 
						// 2 gets inflow 
						mats.Face(f,1,0).AddMult(xlocal, ylocal2, -dot); 
					} 

					else { // Omega points 2 -> 1 
						mats.Face(f,0,1).AddMult(xlocal2, ylocal, dot); 
						mats.Face(f,1,1).AddMult(xlocal2, ylocal2, -dot); 
					}
					for (int i=0; i<vdofs2.Size(); i++) { yview(g,a,vdofs2[i]) += ylocal2(i); }
				}

				else {
					if (dot >= 0) {
						mats.Face(f,0,0).AddMult(xlocal, ylocal, dot); 
					} 
				}
				for (int i=0; i<vdofs.Size(); i++) { yview(g,a,vdofs[i]) += ylocal(i); }
			}
		}
	}
}

void ColumnSum(const mfem::DenseMatrix &mat, mfem::Vector &sums)
{
	sums.SetSize(mat.Width());
	for (int c=0; c<mat.Width(); c++) {
		sums(c) = 0.0;
		for (int r=0; r<mat.Height(); r++) {
			sums(c) += mat(r,c);
		}
	}
}

double AdvectionOperator::ComputeBalance(const mfem::Vector &psi) const
{
	auto &mesh = Linv.mesh;
	const auto dim = mesh.Dimension(); 
	auto &fes = Linv.fes; 
	const auto &psi_ext = Linv.psi_ext; 
	const auto &psi_fnbr_ext = Linv.psi_fnbr_ext; 
	const auto G = psi_ext.extent(0); 
	const auto &quad = Linv.quad; 
	const auto Nomega = quad.Size(); 
	const auto &sched = Linv.scheduler;

	const auto &mats = Linv.mats;

	TransportVectorView psi_view(psi.GetData(), psi_ext); 

	mfem::DenseMatrix *time_mat = nullptr;
	mfem::Vector psi_local, local, weights, time_weights; 
	mfem::Array<int> dofs;
	double absorption = 0.0;
	for (int e=0; e<fes.GetNE(); e++) {
		const auto &fe = *fes.GetFE(e);
		const auto ndof = fe.GetDof();
		psi_local.SetSize(ndof);
		local.SetSize(ndof);
		fes.GetElementDofs(e, dofs);
		for (int g=0; g<G; g++) {
			auto M = mats.Mass(g,e);
			if (Linv.is_time_dependent) M.Add(Linv.time_absorption, mats.TimeMass(e));
			for (int a=0; a<quad.Size(); a++) {
				for (int n=0; n<ndof; n++) { psi_local(n) = psi_view(g,a,dofs[n]); }
				M.Mult(psi_local, local);
				absorption += local.Sum() * quad.GetWeight(a);
			}
		}
	}

	double leakage = 0.0;
	mfem::Vector nor(dim);
	for (int bf=0; bf<mesh.GetNBE(); bf++) {
		const auto f = mesh.GetBdrElementFaceIndex(bf);
		int e, info;
		mesh.GetBdrElementAdjacentElement(bf, e, info);
		fes.GetElementDofs(e, dofs);
		psi_local.SetSize(dofs.Size()); 
		local.SetSize(dofs.Size());
		sched.OutwardNormal(f, 0, nor);
		for (int a=0; a<quad.Size(); a++) {
			const auto &Omega = quad.GetOmega(a);
			const auto dot = Omega*nor;
			if (dot >= 0) {
				for (int g=0; g<G; g++) {
					for (int n=0; n<dofs.Size(); n++) { psi_local(n) = psi_view(g,a,dofs[n]); }
					mats.Face(f,0,0).Mult(psi_local, local); 
					leakage += quad.GetWeight(a) * dot * local.Sum();
				}				
			}
		}
	}
	return leakage + absorption;
}