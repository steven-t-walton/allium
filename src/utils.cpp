#include "utils.hpp"
#include "metis.h"

namespace utils
{

unsigned int Floor(mfem::Vector &x, double min)
{
	unsigned int count = 0;
	for (auto &val : x) {
		if (val <= min) {
			val = min; 
			count++;
		}
	}
	return count;
}

int *GenerateMetisPartitioning(mfem::Mesh &mesh, int nparts, mfem::Coefficient &coef, int method)
{
	if (nparts == 1) return nullptr;
	auto Ne = mesh.GetNE();
	auto connectivity = mesh.ElementToElementTable();
	auto *I = connectivity.GetI(); 
	auto *J = connectivity.GetJ();
	int *partitioning = new int[Ne];
	int ncon = 1;
	int errflag;
	int options[40];
	int edgecut;
	METIS_SetDefaultOptions(options);
	options[METIS_OPTION_CONTIG] = 1; // set METIS_OPTION_CONTIG
	for (int i = 0; i < Ne; i++) {
		std::sort(J+I[i], J+I[i+1], std::greater<int>());
	}

	if (method < 0 or method > 3) MFEM_ABORT("bad method");

	mfem::Array<int> edge_weights_int(connectivity.Size_of_connections());
	if (method & 1) {
		mfem::Vector edge_weights(connectivity.Size_of_connections());
		int idx = 0; 
		for (int i=0; i<connectivity.Size(); i++) {
			auto *row = connectivity.GetRow(i);
			auto &trans_from = *mesh.GetElementTransformation(i);
			auto vfrom = coef.Eval(trans_from, mfem::Geometries.GetCenter(trans_from.GetGeometryType()));
			auto hfrom = mesh.GetElementSize(i);
			for (int j=0; j<connectivity.RowSize(i); j++) {
				auto &trans_to = *mesh.GetElementTransformation(row[j]);
				auto vto = coef.Eval(trans_to, mfem::Geometries.GetCenter(trans_to.GetGeometryType()));
				auto hto = mesh.GetElementSize(row[j]);
				edge_weights(idx++) = std::exp(-vfrom*hfrom/2 - vto*hto/2);
			}
		}
		auto scale = 1e4;
		for (int i=0; i<edge_weights.Size(); i++) {
			edge_weights_int[i] = std::round(edge_weights(i) * scale)+1;
		}
	} else {
		edge_weights_int = 1;
	}

	mfem::Array<int> node_weights(Ne);
	if (method & 2) {
		for (int e=0; e<Ne; e++) {
			auto &trans = *mesh.GetElementTransformation(e);
			auto val = coef.Eval(trans, mfem::Geometries.GetCenter(trans.GetGeometryType()));
			node_weights[e] = std::round(val) + 1;
		}		
	} else {
		node_weights = 1;
	}

	METIS_PartGraphKway(&Ne, &ncon, I, J, node_weights.GetData(), NULL, edge_weights_int.GetData(), &nparts, 
		NULL, NULL, options, &edgecut, partitioning);
	return partitioning;
}

}