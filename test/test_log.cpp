#include "gtest/gtest.h"
#include "mfem.hpp"
#include "log.hpp"
#include <map>
#include <variant>

TEST(EventLog, Synchonize) {
	const auto np = mfem::Mpi::WorldSize(); 
	const auto rank = mfem::Mpi::WorldRank(); 
	EventLog["world size"] = 1; 
	if (rank%2) EventLog["only on even"] = 1; 
	EventLog.Synchronize(); 
	if (mfem::Mpi::Root()) {
		EXPECT_EQ(EventLog["world size"], np); 
		EXPECT_EQ(EventLog["only on even"], np/2); 
	}
}

TEST(EventLog, LogValue) {
	EventLog.clear();
	EventLog.Register("test");
	EventLog.Register("test");
	EventLog.Register("test");
	EXPECT_EQ(EventLog["test"], 3);
}

TEST(TimingLog, Sum) {
	TimingLog.Log("test", 5.0);
	TimingLog.Log("test", 4.0);
	std::cout << TimingLog["test"] << std::endl; 
}

template<typename T>
class Dictionary 
{
private:
	std::map<std::string,std::unique_ptr<Dictionary>> map;
	T data; 
public:
	Dictionary &operator[](std::string str) {
		auto it = map.find(str); 
		Dictionary *dict_ptr;
		if (it == map.end()) {
			auto uptr = std::make_unique<Dictionary>(); 
			dict_ptr = uptr.get(); 
			map[str] = std::move(uptr); 
		} else {
			dict_ptr = it->second.get(); 
		}
		return *dict_ptr; 
	}
	T &operator*() { return data; }
	T *operator->() { return &data; }
	void operator=(T value) { data = value; }
};

TEST(IterLog, test) {
	Dictionary<int> dct; 
	dct["test"] = 5; 
	std::cout << "value = " << *dct["test"] << std::endl; 
	Dictionary<mfem::Array<int>> array; 
	array->SetSize(5); 
}