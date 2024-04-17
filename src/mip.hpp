#pragma once 
#include "mfem.hpp"

class MIPDiffusionIntegrator : public mfem::BilinearFormIntegrator
{
protected:
   mfem::Coefficient *Q;
   mfem::MatrixCoefficient *MQ;
   double sigma, kappa, alpha=0.25;
   bool lump = false; 

   // these are not thread-safe!
   mfem::Vector shape1, shape2, dshape1dn, dshape2dn, nor, nh, ni;
   mfem::DenseMatrix jmat, dshape1, dshape2, mq, adjJ;

public:
   MIPDiffusionIntegrator(const double s, const double k)
      : Q(NULL), MQ(NULL), sigma(s), kappa(k) { }
   MIPDiffusionIntegrator(mfem::Coefficient &q, const double s, const double k, const double a, bool lump=false)
      : Q(&q), MQ(NULL), sigma(s), kappa(k), alpha(a) { }
   MIPDiffusionIntegrator(mfem::MatrixCoefficient &q, const double s, const double k)
      : Q(NULL), MQ(&q), sigma(s), kappa(k) { }
   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const mfem::FiniteElement &el1,
                                   const mfem::FiniteElement &el2,
                                   mfem::FaceElementTransformations &Trans,
                                   mfem::DenseMatrix &elmat);
};

