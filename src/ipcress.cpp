#include "ipcress.hpp"
#include "mdspan/mdspan.hpp"

// place helper functions in annoymous namespace 
// to prevent possible linker issues if functions
// with the same signature happen to be defined 
// elsewhere 
namespace {

bool is_big_endian() {
	union {
		uint32_t i;
		std::array<char, 4> c;
	} data = {0x01020304};

	return data.c[0] == 1;
}

bool big_endian = is_big_endian();

inline void char_byte_swap(unsigned char *data, int n) {
	unsigned char *end = data + n - 1;
	while (data < end)
		std::swap(*data++, *end--);
}

inline void char_byte_swap(char *data, int n) {
	char *end = data + n - 1;
	while (data < end)
		std::swap(*data++, *end--);
}

template <typename T> void byte_swap(T &value) {
	char_byte_swap((unsigned char *)(&value), sizeof(T));
}

template<typename Container>
void read(std::ifstream &inp, std::size_t byte_loc, Container &data)
{
	using ValueType = Container::value_type;
	std::vector<char> buffer(data.size()*8); 

	inp.seekg(byte_loc, std::ios::beg);
	inp.read(buffer.data(), buffer.size());

	if constexpr (std::is_same<ValueType,std::string>::value) {
		for (int i=0; i<data.size(); i++) {
			data[i] = std::string(&buffer[8*i], 8);
		}		
	} else {
		for (int i=0; i<data.size(); i++) {
			double check; 
			memcpy(&check, &buffer[i*8], 8);
			if (!big_endian)
				byte_swap(check);
			data[i] = static_cast<ValueType>(check);
		}		
	}
}

template<typename T>
T read(std::ifstream &inp, std::size_t byte_loc)
{
	std::vector<char> buffer(8);
	inp.seekg(byte_loc, std::ios::beg);
	inp.read(buffer.data(), buffer.size()); 
	double check; 
	memcpy(&check, buffer.data(), 8);
	if (!big_endian)
		byte_swap(check);
	return static_cast<T>(check);
}

std::string read_string(std::ifstream &inp, std::size_t byte_loc)
{
	std::vector<char> buffer(8);
	inp.seekg(byte_loc, std::ios::beg);
	inp.read(buffer.data(), buffer.size()); 
	return std::string(buffer.data(), 8);
}

}

IpcressData::IpcressData(const std::string &file_name)
	: file_name(file_name)
{
	std::ifstream inp(file_name, std::ios::binary | std::ios::in);
	const auto title = read_string(inp, 0); 
	if (title != "nirvana ") MFEM_ABORT("title = " << title);

	std::vector<std::size_t> toc(24);
	read(inp, 16, toc);
	if (toc[2] != 24 or toc[4] != 2 or toc[5] != 0) MFEM_ABORT("wrong toc");

	const auto num_records = toc[14]; 
	std::vector<std::size_t> dfo(num_records), ds(num_records);
	read(inp, toc[0]*8, ds);
	read(inp, toc[1]*8, dfo);

	const auto num_mats = read<int>(inp, dfo[0]*8); 

	std::vector<std::size_t> matIds(num_mats);
	read(inp, 8*(dfo[0]+1), matIds);

	data.resize(std::accumulate(ds.begin()+1, ds.end(), 0));
	data_view.resize(num_mats); 

	std::size_t loc = toc[10] * 8;
	std::vector<std::string> sub_string(3);
	std::size_t part_sum = 0;
	for (auto i=1; i<num_records; i++) {
		read(inp, loc + 9 * i * 8, sub_string);
		std::string field = sub_string[0] + sub_string[1] + sub_string[2];
		const auto id = atoi(field.c_str());
		auto it = std::find(matIds.begin(), matIds.end(), id);
		std::size_t matid;
		if (it != matIds.end()) {
			matid = std::distance(matIds.begin(), it);
		} else { MFEM_ABORT("not found"); }
		read(inp, loc + 3 * (3 * i + 1) * 8, sub_string);
		field = sub_string[0] + sub_string[1] + sub_string[2];
		field.erase(std::remove_if(field.begin(), field.end(), ::isspace), field.end());
		auto sub = std::span<double>(data.begin()+part_sum, ds[i]);
		data_view[matid][field] = sub;
		read(inp, dfo[i]*8, sub);
		part_sum += ds[i];
	}

	std::size_t num_groups = data_view[0].at("hnugrid").size();
	for (int i=0; i<num_mats; i++) {
		auto &temperature = data_view[i].at("tgrid"); 
		for (auto &value : temperature) { value *= 1e3; }
		auto &hnugrid = data_view[i].at("hnugrid");
		if (num_groups != hnugrid.size()) MFEM_ABORT("mixed number of groups in materials");
		for (auto &value : hnugrid) { value *= 1e3; }
		for (const auto &key : {"tgrid", "rgrid", "ramg", "pmg", "rgray", "pgray"}) {
			if (data_view[i].find(key) != data_view[i].end()) {
				auto &sub = data_view[i].at(key);
				std::transform(sub.begin(), sub.end(), sub.begin(), [](const double x) { return std::log(x); });				
			}
		}
	}

	auto &grid = data_view[0].at("hnugrid"); 
	bounds.MakeRef(grid.data(), grid.size());

	inp.close();
}

void IpcressData::Eval(int matid, const std::string &key, double rho, double T, mfem::Vector &v) const 
{
	const bool is_gray = key.size() >= 4 and key.substr(key.size()-4) == "gray"; 
	auto density = GetField(matid, "rgrid"); 
	auto temperature = GetField(matid, "tgrid");
	auto value = GetField(matid, key);

	rho = std::log(rho); 
	T = std::log(T);

	const auto Tid = FindIndex(temperature, T);
	const auto rid = FindIndex(density, rho);

	auto xi = (T - temperature[Tid]) / (temperature[Tid+1] - temperature[Tid]); 
	auto eta = (rho - density[rid]) / (density[rid+1] - density[rid]); 
	assert(xi >= 0.0 and xi <= 1.0); 
	assert(eta >= 0.0 and eta <= 1.0);
	std::array<double,4> basis = {(1.0-xi)*(1.0-eta), xi*(1.0-eta), (1.0-xi)*eta, xi*eta}; 

	const auto G = (is_gray) ? 1 : bounds.Size() - 1;
	using Extents = Kokkos::dextents<std::size_t,3>; 
	auto value_view = Kokkos::mdspan<const double,Extents,Kokkos::layout_left>(
		value.data(), G, density.size(), temperature.size());
	v.SetSize(G); 
	for (int g=0; g<G; g++) {
		const double log_opac = 
			value_view(g,rid,Tid) * basis[0]
			+ value_view(g,rid,Tid+1) * basis[1]
			+ value_view(g,rid+1,Tid) * basis[2] 
			+ value_view(g,rid+1,Tid+1) * basis[3];

		v(g) = std::exp(log_opac);
	}
}

double IpcressData::Eval(int matid, const std::string &key, double rho, double T, double E) const
{
	auto density = GetField(matid, "rgrid"); 
	auto temperature = GetField(matid, "tgrid");
	auto bounds = GetField(matid, "hnugrid");
	auto value = GetField(matid, key);

	rho = std::log(rho); 
	T = std::log(T);

	const auto group = FindIndex(bounds, E);
	const auto Tid = FindIndex(temperature, T);
	const auto rid = FindIndex(density, rho);

	auto xi = (T - temperature[Tid]) / (temperature[Tid+1] - temperature[Tid]); 
	auto eta = (rho - density[rid]) / (density[rid+1] - density[rid]); 
	assert(xi >= 0.0 and xi <= 1.0); 
	assert(eta >= 0.0 and eta <= 1.0);
	std::array<double,4> basis = {(1.0-xi)*(1.0-eta), xi*(1.0-eta), (1.0-xi)*eta, xi*eta}; 

	const auto G = bounds.size() - 1;
	using Extents = Kokkos::dextents<std::size_t,3>; 
	Extents ext(G, density.size(), temperature.size());
	auto value_view = Kokkos::mdspan<const double,Extents,Kokkos::layout_left>(value.data(), ext);
	const double log_opac = 
		value_view(group,rid,Tid) * basis[0]
		+ value_view(group,rid,Tid+1) * basis[1]
		+ value_view(group,rid+1,Tid) * basis[2] 
		+ value_view(group,rid+1,Tid+1) * basis[3];
	return std::exp(log_opac);
}

std::size_t IpcressData::FindIndex(std::span<const double> &field, double &value) const 
{
	std::size_t id = -1;
	if (value <= field.front()) {
		value = field.front(); 
		id = 0;
	}
	else if (value >= field.back()) {
		value = field.back();
		id = field.size()-2;
	} else {
		auto it = std::lower_bound(field.begin(), field.end(), value);
		id = std::distance(field.begin(), it)-1;
	}
	assert(id >= 0 and id < field.size()-1);
	return id;
}