#pragma once 

#include "opacity.hpp"
#include "mg_form.hpp"
#include "sweep.hpp"
#include "log.hpp"

class OpacityUpdate
{
private:
	ProjectedVectorCoefficient &total;
	MultiGroupBilinearForm &Mtot;
	InverseAdvectionOperator &Linv;
	mfem::StopWatch timer;
public:
	OpacityUpdate(ProjectedVectorCoefficient &total, MultiGroupBilinearForm &Mtot, 
		InverseAdvectionOperator &Linv)
		: total(total), Mtot(Mtot), Linv(Linv) 
	{ }
	void Update()
	{
		timer.Restart();
		total.Project();
		TimingLog.Log("opacity", timer.RealTime());
		timer.Restart();
		Mtot.Assemble();
		Mtot.Finalize();
		TimingLog.Log("assemble Mtot", timer.RealTime());
		Linv.AssembleLocalMatrices();
	}
};
