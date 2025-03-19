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
	auto Ne = mesh.GetNE(); // number of elements 
	auto connectivity = mesh.ElementToElementTable();
	auto *I = connectivity.GetI(); // ptr to offsets array 
	auto *J = connectivity.GetJ(); // ptr to column indices 
	int *partitioning = new int[Ne]; // return data, assigns [0,nparts) to each element 
	int ncon = 1; // "number of constraints". 1 means just balance volume to surface ratio 
	int errflag; // return flag 
	int options[40]; // I guess there are 40 possible options... 
	int edgecut; // return value from metis that measures the communication surface area of the partition 
	METIS_SetDefaultOptions(options); // fills the 40 options to defaults 
	options[METIS_OPTION_CONTIG] = 1; // set METIS_OPTION_CONTIG: tell metis to make contiguous partitions 

	// not sture if this is necessary but MFEM does this
	// sorts each row of the connectivity from small to large 
	for (int i = 0; i < Ne; i++) {
		std::sort(J+I[i], J+I[i+1], std::greater<int>());
	}

	// 0: default metis 
	// 1: edge weight 
	// 2: node weight 
	// 3: edge and node weight 
	if (method < 0 or method > 3) MFEM_ABORT("bad method");

	// edge weights must be integers... 
	mfem::Array<int> edge_weights_int(connectivity.Size_of_connections());
	// compute edge weight from opacity data 
	if (method & 1) {
		mfem::Vector edge_weights(connectivity.Size_of_connections());
		int idx = 0; 
		for (int i=0; i<connectivity.Size(); i++) {
			auto *row = connectivity.GetRow(i); // neighboring elements for element i 
			// compute cell centers of tip and tail of edge 
			auto &trans_from = *mesh.GetElementTransformation(i); 
			auto vfrom = coef.Eval(trans_from, mfem::Geometries.GetCenter(trans_from.GetGeometryType()));
			auto hfrom = mesh.GetElementSize(i);
			for (int j=0; j<connectivity.RowSize(i); j++) {
				auto &trans_to = *mesh.GetElementTransformation(row[j]);
				auto vto = coef.Eval(trans_to, mfem::Geometries.GetCenter(trans_to.GetGeometryType()));
				auto hto = mesh.GetElementSize(row[j]);
				// edge_weights(idx++) = std::exp(-vfrom*hfrom/2 - vto*hto/2);
				edge_weights(idx++) = 1.0 / (vfrom*hfrom/2 + vto*hto/2); 
			}
		}
		// scale to map to integers better 
		auto scale = 1e4;
		for (int i=0; i<edge_weights.Size(); i++) {
			edge_weights_int[i] = std::round(edge_weights(i) * scale)+1;
		}
	} 

	// otherwise set to default value 
	else {
		edge_weights_int = 1;
	}

	// compute node weights
	// must be integer... 
	mfem::Array<int> node_weights(Ne);
	if (method & 2) {
		for (int e=0; e<Ne; e++) {
			auto &trans = *mesh.GetElementTransformation(e);
			auto val = coef.Eval(trans, mfem::Geometries.GetCenter(trans.GetGeometryType()));
			node_weights[e] = std::round(val) + 1;
		}		
	} 

	// otherwise set to default/uniform value 
	else {
		node_weights = 1;
	}

	// call metis API 
	// Ne: number of elements 
	// ncon: number of constraints (should be 1 for standard usage)
	// I: CSR offset array for connectivity (neighbors per element in offset format)
	// J: CSR column array for connectivity (list of neighbors for each element)
	// node_weights: integer array size of number of elements 
	// NULL: relates to number of vertices (it is optional, I think Metis computes this itself if not provided)
	// edge_weights: integer array size of number of edges 
	// nparts: number of partitions to create 
	// NULL: relates to extra constraints when ncon>1 
	// NULL: weights to use for each constraint when ncon>1 
	// options: integer array of configurable options 
	// edgecut: output parameter measuring how good the partition is 
	// partitioning: integer pointer size of number of elements with a partition for each element 
	METIS_PartGraphKway(&Ne, &ncon, I, J, node_weights.GetData(), NULL, edge_weights_int.GetData(), &nparts, 
		NULL, NULL, options, &edgecut, partitioning);
	return partitioning;
}

InterpolatedTable1D::InterpolatedTable1D(const std::string &file_name)
{
	std::ifstream inp(file_name);
	if (!inp.good()) { MFEM_ABORT("file not opened"); }

	std::string line; 
	int count=0;
	while (std::getline(inp, line)) {
		count++;
	}

	x.SetSize(count); 
	y.SetSize(count);

	inp.clear();
	inp.seekg(0, std::ios::beg);
	for (int i=0; i<count; i++) {
		inp >> x[i] >> y[i];
	}
	inp.close();
}

double InterpolatedTable1D::Eval(double val) const
{
	const auto N = x.Size();
	if (val <= x[0]) {
		return y[0];
	} else if (val >= x[N-1]) {
		return y[N-1];
	} 

	auto it = std::lower_bound(x.begin(), x.end(), val);
	std::size_t loc = std::distance(x.begin(), it)-1;

	assert(loc>=0 and loc < x.Size()-1);
	assert(val >= x[0] and val <= x[N-1]);

	if (piecewise_constant) return y[loc]; 

	std::array<double,2> xvals = {x[loc], x[loc+1]};
	std::array<double,2> yvals = {y[loc], y[loc+1]};
	if (log_x) {
		std::transform(xvals.begin(), xvals.end(), xvals.begin(), 
			[](const double v) { return std::log(v); });
		val = std::log(val);
	}
	if (log_y) {
		std::transform(yvals.begin(), yvals.end(), yvals.begin(), 
			[](const double v) { return std::log(v); });		
	}
	const double xi = (val - xvals[0]) / (xvals[1] - xvals[0]);
	assert(xi >= 0.0 and xi <= 1.0);
	std::array<double,2> shape = {1.0-xi, xi};
	const double r = shape[0] * yvals[0] + shape[1] * yvals[1];
	if (log_y) return std::exp(r);
	else return r;
}

}