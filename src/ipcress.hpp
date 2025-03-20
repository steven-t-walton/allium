#pragma once 

#include "mfem.hpp"
#include <span>

class IpcressData 
{
private:
	const std::string file_name;
	std::vector<double> data; 
	std::vector<std::unordered_map<std::string,std::span<double>>> data_view;
	mfem::Array<double> bounds;
public:
	IpcressData(const std::string &file_name); 
	void Eval(int matid, const std::string &key, double rho, double temperature, mfem::Vector &v) const;
	double Eval(int matid, const std::string &key, double rho, double temperature, double E) const;
	const std::string &FileName() const { return file_name; }
	const mfem::Array<double> &GetGroupBounds() const { return bounds; }
	int NumGroups() const { return bounds.Size() - 1; }
	int NumMaterials() const { return data_view.size(); }
	std::span<const double> GetField(int matid, const std::string &key) const
	{
		return data_view.at(matid).at(key);
	}
private:
	std::size_t FindIndex(std::span<const double> &field, double &value) const;
};