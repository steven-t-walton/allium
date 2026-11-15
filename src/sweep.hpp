#pragma once

#include "bdr_conditions.hpp"
#include "tvector.hpp"
#include "angular_quadrature.hpp"
#include "mfem.hpp"
#include "phase_coefficient.hpp"
#include "fixup.hpp"
#include "dg_local_matrices.hpp"
#include "sweep_scheduler.hpp"

class MultiGroupEnergyGrid; // forward declare to prevent compile dependency
class MultiGroupCoefficient;

// multigroup Sn sweep kernel: dense element-local DG solve for each ordinate and group
// angular flux layout psi(g,a,dof) given by TransportVectorView
// face neighbor data is stored in psi_fnbr with index a + g*Nomega + Nomega*G*dof
class DenseSnKernel : public SweepKernel
{
private:
	mfem::ParFiniteElementSpace &fes;
	const AngularQuadrature &quad;
	const DGLocalMatrices &mats;
	const SweepScheduler &sched;
	TransportVectorExtents psi_ext;
	const double *source = nullptr;
	double *psi = nullptr;
	double *psi_fnbr = nullptr;

	// persistent per-thread storage of the local system
	struct Scratch {
		mfem::DenseMatrix grad, A, lu;
		mfem::Vector rhs, sol, psi2, nor;
		mfem::Array<int> dofs, dofs2;
		int fixup_success = 0, fixup_failed = 0;
	};
	std::vector<Scratch> scratch;

	int GhostIndex(int g, int a, int dof) const {
		return a + g*psi_ext.extent(1) + psi_ext.extent(1)*psi_ext.extent(0)*dof;
	}
public:
	// time absorption (e.g. 1/c/dt) added to the diagonal when time_dependent
	bool time_dependent = false;
	double time_absorption = 0.0;
	const NegativeFluxFixupOperator *fixup_op = nullptr;
	mfem::Vector *fixup_monitor = nullptr;

	DenseSnKernel(mfem::ParFiniteElementSpace &fes, const AngularQuadrature &quad,
		const DGLocalMatrices &mats, const SweepScheduler &sched);
	// set the vectors used by the next sweep, source and psi may alias
	void Bind(const mfem::Vector &source, mfem::Vector &psi, mfem::Vector &psi_fnbr);
	// log and reset fixup counters
	void LogFixup();

	void Solve(int e, int set, std::span<const SweepEdge> in) override;
	std::size_t PackSize(int e, int set) const override;
	void Pack(int e, int set, char *buf) const override;
	void Unpack(int ghost_e, int set, const char *buf) override;
};

class InverseAdvectionOperator : public mfem::Operator
{
protected:
	mfem::ParMesh &mesh;
	mfem::ParFiniteElementSpace &fes;
	const AngularQuadrature &quad;
	MultiGroupCoefficient &total;

	// multi index extents for angular flux and the "ghost"
	// buffer psi_fnbr
	TransportVectorExtents psi_ext, psi_fnbr_ext;

	// use lumping
	int lump = 0;
	// element and face matrices of the local sweep systems
	DGLocalMatrices mats;
	// graph over (element, angle) and its parallel traversal
	SweepScheduler scheduler;
	mutable DenseSnKernel kernel;

	// time-dependent sweep data
	bool is_time_dependent = false;
	double time_absorption;

	// parallel buffer for psi
	mutable mfem::Vector psi_fnbr;

	// retained for the input interface, messages are sent once per wave front
	int send_buffer_size = 8;
	int max_sends_per_recv = 8;

	bool apply_fixup = false;
	const NegativeFluxFixupOperator *fixup_op = nullptr;
	mfem::Vector *fixup_monitor = nullptr;

	bool pbj = false;
	int num_pbj_sweeps = 1;

	// halo exchange of psi into psi_fnbr
	void ExchangeGhost(const mfem::Vector &psi) const;
public:
	InverseAdvectionOperator(mfem::ParFiniteElementSpace &_fes, const AngularQuadrature &_quad,
		MultiGroupCoefficient &total, const BoundaryConditionMap &bc_map, int lump=0);

	void Mult(const mfem::Vector &source, mfem::Vector &psi) const;
private:
	void Mult_Upwind(const mfem::Vector &source, mfem::Vector &psi) const;
	void Mult_BlockJacobi(const mfem::Vector &source, mfem::Vector &psi) const;
	void SetupKernel(const mfem::Vector &source, mfem::Vector &psi) const;
public:
	// load the psi_fnbr data structure from psi
	// only needed for PBJ where psi_fnbr represents
	// the initial guess for lagged parallel data
	void Exchange(const mfem::Vector &psi) { ExchangeGhost(psi); }

	// must be called before Mult
	// pre-assembles components of local sweep matrix
	// to save cost
	void AssembleLocalMatrices() { mats.Assemble(); }
	// indicates sweep is time dependent with
	// provided time absorption (e.g. 1/c/dt)
	// time mass matrix is assembled once and stored
	void SetTimeAbsorption(const double sigma);
	// access set time absorption value
	double GetTimeAbsorption() const { return time_absorption; }

	// use parallel block jacobi sweep
	void UseParallelBlockJacobi(bool use=true) { pbj = use; }
	bool IsParallelBlockJacobi() const { return pbj; }
	// how many PBJ sweeps to do per call to Mult
	void SetNumPBJSweeps(int n) { num_pbj_sweeps = n; }

	void SetSendBufferSize(int s) { send_buffer_size = s; }
	void SetMaxSendsPerReceive(int s) { max_sends_per_recv = s; }
	// allow disabling or re-enabling fixup
	void UseFixup(bool use=true);
	bool IsFixupOn() const { return apply_fixup and fixup_op; }
	void SetFixupOperator(const NegativeFluxFixupOperator &op) {
		fixup_op = &op;
		apply_fixup = true;
	}
	void SetFixupMonitorData(mfem::Vector &x) { fixup_monitor = &x; }
	void WriteGraphToDot(std::string prefix) const { scheduler.WriteGraphToDot(prefix); }
	void WriteGlobalGraphToDot(std::string prefix) const;

	// lumping type accessors
	int GetLumpingType() const { return lump; }

	const DGLocalMatrices &GetLocalMatrices() const { return mats; }
	const SweepScheduler &GetScheduler() const { return scheduler; }

	friend class AdvectionOperator;
};

void FormTransportSource(mfem::ParFiniteElementSpace &fes, AngularQuadrature &quad,
	const mfem::Array<double> &energy_grid, PhaseSpaceCoefficient &source_coef,
	PhaseSpaceCoefficient &inflow_coef, TransportVectorView source_view);
void FormTransportSource(
	mfem::FiniteElementSpace &fes, const AngularQuadrature &quad,const MultiGroupEnergyGrid &energy_grid,
	PhaseSpaceCoefficient &source_coef, PhaseSpaceCoefficient &inflow_coef,
	mfem::Vector &source);

// apply the transport streaming and collision operator
// performs a halo exchange of all angles, groups
// uses assembled matrices from InverseAdvectionOperator
class AdvectionOperator : public mfem::Operator
{
private:
	const InverseAdvectionOperator &Linv;
public:
	AdvectionOperator(const InverseAdvectionOperator &Linv) : Linv(Linv)
	{ }
	void Mult(const mfem::Vector &x, mfem::Vector &y) const override;
	double ComputeBalance(const mfem::Vector &psi) const;
};
