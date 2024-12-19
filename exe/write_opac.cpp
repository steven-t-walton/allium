#include "mfem.hpp"
#include "opacity.hpp"
#include "multigroup.hpp"
#include "sol/sol.hpp"
#include "yaml-cpp/yaml.h"
#include "io.hpp"

int main(int argc, char *argv[]) {
	std::string input_file;
	double temperature;
	int N = 100;
	mfem::OptionsParser args(argc, argv); 
	args.AddOption(&input_file, "-i", "--input", "input file name", true); 
	args.AddOption(&temperature, "-t", "--temperature", "temperature to evaluate opacity", true);
	args.AddOption(&N, "-n", "--numpoints", "number of points to plot");
	args.Parse(); 
	if (!args.Good()) {
		args.PrintUsage(std::cout); 
		return 1; 
	}

	YAML::Emitter out(std::cout);
	out.SetDoublePrecision(16); 
	out << YAML::BeginMap; 

	out << YAML::Key << "stefan boltzmann" << YAML::Value << io::FormatScientific(constants::StefanBoltzmann, 16);
	out << YAML::Key << "temperature" << YAML::Value << temperature;

	// --- load lua file --- 
	sol::state lua; 
	lua.open_libraries(); // allows using standard libraries (e.g. math) in input
	lua.script_file(input_file); // load from first cmd line argument 

	sol::optional<std::string> ipcress_path_avail = lua["ipcress_file"];
	std::unique_ptr<IpcressData> ipcress_data;
	if (ipcress_path_avail) {
		const std::string file_name = ipcress_path_avail.value();
		ipcress_data = std::make_unique<IpcressData>(file_name);
		io::PrintIpcressInformation(out, *ipcress_data);
	}

	out << YAML::Key << "energy" << YAML::Value << YAML::BeginMap; 
	MultiGroupEnergyGrid energy_grid;
	if (ipcress_data) {
		energy_grid = MultiGroupEnergyGrid(ipcress_data->GetGroupBounds());
		out << YAML::Key << "groups" << YAML::Value << energy_grid.Size(); 
		out << YAML::Key << "bounds" << YAML::Value;
		io::PrintArray(out, energy_grid.Bounds()); 
	}
	else {
		sol::table energy_table = lua["energy"];
		if (!energy_table.valid()) MFEM_ABORT("must supply energy table");
		energy_grid = io::CreateEnergyGrid(energy_table, out);		
	}
	out << YAML::EndMap; // end energy block 

	double min_energy = energy_grid.MinEnergy();
	if (min_energy == 0.0) min_energy = energy_grid.Bounds()[1];
	auto print_grid = MultiGroupEnergyGrid::MakeLogSpaced(min_energy, energy_grid.MaxEnergy(), N, false);
	const auto &energies = print_grid.Bounds();
	out << YAML::Key << "energies" << YAML::Value << YAML::BeginSeq; 
	for (const auto &E : energies) {
		out << E;
	}
	out << YAML::EndSeq;

	sol::table materials = lua["materials"];
	out << YAML::Key << "materials" << YAML::Value << YAML::BeginMap;
	io::OpacityFactory opac_fact(energy_grid, out, true);
	if (ipcress_data) opac_fact.SetIpcressData(*ipcress_data);
	for (const auto &material : materials) {
		auto key = material.first.as<std::string>(); 
		sol::table data = material.second;
		sol::table total = data["total"];
		out << YAML::Key << key << YAML::Value << YAML::BeginMap;
		auto *opacity = opac_fact.CreateOpacity(total);
		const double density = data["density"];
		out << YAML::Key << "density" << YAML::Value << density;

		out << YAML::Key << "values" << YAML::Value << YAML::BeginSeq;
		for (int i=0; i<energies.Size(); i++) {
			const auto E = energies[i]; 
			const auto val = opacity->Eval(density, temperature, E);
			out << val;
		}
		out << YAML::EndSeq;

		out << YAML::Key << "groups" << YAML::Value << YAML::BeginSeq;
		mfem::Vector vals; 
		mfem::IsoparametricTransformation trans; 
		trans.SetIdentityTransformation(mfem::Geometry::SEGMENT);
		mfem::IntegrationPoint ip;
		mfem::ConstantCoefficient T(temperature), rho(density);
		opacity->SetTemperature(T); opacity->SetDensity(rho);
		opacity->Eval(vals, trans, ip);
		for (int g=0; g<vals.Size(); g++) {
			out << vals(g);
		}
		out << YAML::EndSeq;
		out << YAML::EndMap;
		delete opacity;
	}
	out << YAML::EndMap;

	out << YAML::EndMap << YAML::Newline;
}