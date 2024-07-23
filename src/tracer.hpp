#pragma once 

#include "mfem.hpp"

class TracerDataCollection : public mfem::DataCollection 
{
private:
	std::vector<std::ofstream> outs; 
	mfem::DenseMatrix point_mat; 
	mfem::Array<int> elem_ids; 
	mfem::Array<mfem::IntegrationPoint> ips; 
	bool first = true, restart_mode = false;
public:
	TracerDataCollection(const std::string &collection_name, mfem::Mesh &mesh, const mfem::DenseMatrix &point_mat); 
	~TracerDataCollection(); 
	void SetMesh(mfem::Mesh *mesh) override; 
	void Save() override; 
	void UseRestartMode(bool use=true) { restart_mode = use; }
};