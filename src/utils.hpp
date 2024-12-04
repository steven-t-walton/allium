#pragma once 
#include "mfem.hpp"

namespace utils
{

unsigned int Floor(mfem::Vector &x, double min=0.0);

int *GenerateMetisPartitioning(mfem::Mesh &mesh, int nparts, mfem::Coefficient &coef, int method=3); 

} // end namespace utils 
