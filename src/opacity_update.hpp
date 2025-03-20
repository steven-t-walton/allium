#pragma once 

#include "opacity.hpp"
#include "mg_form.hpp"
#include "sweep.hpp"

class OpacityUpdate
{
private:
	ProjectedVectorCoefficient &total;
	MultiGroupBilinearForm &Mtot;
	InverseAdvectionOperator &Linv;
public:
	OpacityUpdate(ProjectedVectorCoefficient &total, MultiGroupBilinearForm &Mtot, 
		InverseAdvectionOperator &Linv)
		: total(total), Mtot(Mtot), Linv(Linv) 
	{ }
	void Update()
	{
		total.Project();
		Mtot.Assemble();
		Mtot.Finalize();
		Linv.AssembleLocalMatrices();
	}
};
