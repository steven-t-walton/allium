#include "tracer.hpp"

TracerDataCollection::TracerDataCollection(const std::string &collection_name, 
	mfem::Mesh &mesh, const mfem::DenseMatrix &point_mat)
	: mfem::DataCollection(collection_name), point_mat(point_mat)
{
	SetMesh(&mesh); 
}

TracerDataCollection::~TracerDataCollection()
{
	for (auto &out : outs) {
		out.close(); 
	}
}

void TracerDataCollection::SetMesh(mfem::Mesh *mesh) 
{
	DataCollection::SetMesh(mesh); 
	const int npts = mesh->FindPoints(point_mat, elem_ids, ips, false, nullptr); 
	int num_owned = 0; 
	for (const auto &e : elem_ids) {
		if (e == -1) MFEM_ABORT("tracer point not found"); 
		if (e >= 0) { num_owned++; }
	}
	outs.resize(num_owned); 
}

struct FixedWidthFormatter {
	int precision, width; 
	void operator()(std::ostream &out, int value, bool start=false) {
		if (!start) out << ","; 
		out << std::setw(precision) << std::setfill('0') << value; 
	}
	void operator()(std::ostream &out, double value, bool start=false) {
		if (!start) out << ","; 
		out << std::scientific << std::setprecision(precision) << std::setw(width) << value; 
	}
};

void TracerDataCollection::Save()
{
	if (first) {
		const auto err = create_directory(GetPrefixPath(), GetMesh(), myid);
		int count = 0; 
		for (int n=0; n<elem_ids.Size(); n++) {
			const auto e = elem_ids[n]; 
			if (e >= 0) {
				auto &out = outs[count++]; 
				std::stringstream fname_ss; 
				fname_ss << prefix_path << name << "." << n << ".csv"; 
				if (restart_mode) {
					// open in append mode
					out.open(fname_ss.str(), std::ios::app);
				} else {
					// open a new file 
					// write header 
					out.open(fname_ss.str(), std::ios::out); 
					out << "cycle,time,time step,x,y,z";
					for (auto &field : field_map) {
						if (field.second->VectorDim()>1) {
							out << "," << field.first << "_x,"
								<< field.first << "_y," 
								<< field.first << "_z"; 
						}
						else {
							out << "," << field.first; 						
						}
					} 
					out << std::endl; 					
				}
			}
		}
		first = false; 
	}

	const auto width = precision + 3; 
	FixedWidthFormatter f(precision, width); 

	mfem::Vector location(3), vec_val(3); 
	location = 0.0; 
	int count = 0; 
	for (int n=0; n<elem_ids.Size(); n++) {
		const auto e = elem_ids[n]; 
		if (e < 0) continue; // skip ids owned by another processor 

		auto &out = outs[count++]; 
		f(out, cycle, true); 
		f(out, time); 
		f(out, time_step); 
		point_mat.GetColumn(n, location); 
		location.SetSize(3); 
		for (const auto &x : location) {
			f(out, x); 
		}

		auto &trans = *mesh->GetElementTransformation(e); 
		const auto &ip = ips[n]; 
		for (auto &field : field_map) {
			if (field.second->VectorDim() == 1) {
				mfem::GridFunctionCoefficient coef(field.second); 
				const double val = coef.Eval(trans, ip); 
				f(out, val); 				
			} else {
				mfem::VectorGridFunctionCoefficient coef(field.second); 
				coef.Eval(vec_val, trans, ip); 
				vec_val.SetSize(3); 
				for (const auto &v : vec_val) {
					f(out, v); 
				}
			}
		}
		out << std::endl; // <-- use endl to force flush 
	}
}