#pragma once 

#include "mfem.hpp"

mfem::HypreParMatrix *ElementByElementBlockInverse(const mfem::ParFiniteElementSpace &fes, const mfem::HypreParMatrix &A); 