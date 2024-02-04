#pragma once 

#include "mfem.hpp"

class DGTrace_FECollection : public mfem::FiniteElementCollection
{
private:
	int b_type; 
	int m_type; 
	char dt_name[32]; 
	int dim; 
	mfem::ScalarFiniteElement *Elements[mfem::Geometry::NumGeom]; 
	mfem::ScalarFiniteElement *Tr_Elements[mfem::Geometry::NumGeom];
public:
	DGTrace_FECollection(const int p, const int _dim, const int btype = mfem::BasisType::GaussLegendre, 
		const int map_type = mfem::FiniteElement::VALUE); 

	~DGTrace_FECollection() {
		for (int g=0; g<mfem::Geometry::NumGeom; g++) {
			if (Elements[g]) delete Elements[g]; 
			if (Tr_Elements[g]) delete Tr_Elements[g]; 
		}
	}

	const mfem::FiniteElement *
	FiniteElementForGeometry(mfem::Geometry::Type GeomType) const override {
		return Elements[GeomType]; 
	}
	const mfem::FiniteElement *
	TraceFiniteElementForGeometry(mfem::Geometry::Type GeomType) const override
	{
		return Tr_Elements[GeomType]; 
	}

	int DofForGeometry(mfem::Geometry::Type GeomType) const override {
		if (dim==1 and GeomType == mfem::Geometry::SEGMENT)
		{
			return 2; 
		}
		else if (dim==2 and (GeomType == mfem::Geometry::SQUARE or GeomType == mfem::Geometry::TRIANGLE)) {
			const auto nf = mfem::Geometries.NumBdr(GeomType); 
			return nf*(base_p+1); 						
		} 
		else if (dim==3 and GeomType == mfem::Geometry::CUBE) {
			const auto nf = mfem::Geometries.NumBdr(GeomType); 
			return nf*pow(base_p+1,2); 
		}
		return 0; 
	}

	const int *DofOrderForOrientation(mfem::Geometry::Type GeomType, int Or) const override {
		MFEM_ABORT("not implemented"); 
		return nullptr; 
	}

	const char *Name() const override { return dt_name; }
	int GetContType() const override { return DISCONTINUOUS; }
	int GetBasisType() const { return b_type; }
};