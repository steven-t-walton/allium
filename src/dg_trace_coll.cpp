#include "dg_trace_coll.hpp"

DGTrace_FECollection::DGTrace_FECollection(const int p, const int _dim, const int btype, const int map_type) 
	: dim(_dim), b_type(btype), m_type(map_type), mfem::FiniteElementCollection(p) 
{
	snprintf(dt_name, 32, "DGTrace_%d_%D", p, dim);  
	for (int g=0; g<mfem::Geometry::NumGeom; g++) {
		Tr_Elements[g] = nullptr; 
		Elements[g] = nullptr; 
	}
	if (dim==1) {
		Tr_Elements[mfem::Geometry::POINT] = new mfem::PointFiniteElement; 
		Elements[mfem::Geometry::SEGMENT] = new mfem::L2_SegmentElement(p,btype); 
	} 
	else if (dim==2) {
		Tr_Elements[mfem::Geometry::SEGMENT] = new mfem::L2_SegmentElement(p, btype); 
		Elements[mfem::Geometry::SQUARE] = new mfem::L2_QuadrilateralElement(p,btype); 
	}
	else if (dim==3) {
		Tr_Elements[mfem::Geometry::SQUARE] = new mfem::L2_QuadrilateralElement(p, btype); 
		Tr_Elements[mfem::Geometry::TRIANGLE] = new mfem::L2_TriangleElement(p, btype); 
		Elements[mfem::Geometry::CUBE] = new mfem::L2_HexahedronElement(p, btype); 
	}
}