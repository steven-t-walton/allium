#pragma once 
#include "mfem.hpp"

namespace utils
{

unsigned int Floor(mfem::Vector &x, double min=0.0);

int *GenerateMetisPartitioning(mfem::Mesh &mesh, int nparts, mfem::Coefficient &coef, int method=3); 

class InterpolatedTable1D
{
private:
	mfem::Vector x, y;
	bool log_x, log_y;
	bool piecewise_constant = false;
public:
	InterpolatedTable1D(const std::string &file);
	InterpolatedTable1D(const mfem::Vector &x, const mfem::Vector &y)
		: x(x), y(y)
	{ }
	void UseLogX(bool use=true) { log_x = use; }
	void UseLogY(bool use=true) { log_y = use; }
	void UsePiecewiseConstant(bool use=true) { piecewise_constant = use; }
	double Eval(double val) const;
};

} // end namespace utils 
