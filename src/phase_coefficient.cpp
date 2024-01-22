#include "phase_coefficient.hpp"

double FunctionGrayCoefficient::Eval(mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) {
	double x[3]; 
	mfem::Vector transip(x, 3); 
	trans.Transform(ip, transip); 	
	return f(transip, Omega); 
}