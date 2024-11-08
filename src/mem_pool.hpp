#pragma once 

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPoolList.hpp"
#include "umpire/strategy/QuickPool.hpp"
#include "mfem.hpp"

namespace internal 
{

// singleton class that registers an umpire pool 
// with strategy and name and provides accessor 
// through Get() 
// singleton pattern is used so that 
// makeAllocator is called only once, preventing 
// umpire already defined allocator error 
template<typename strategy, const char name[]>
class UmpirePool
{
public:
	static umpire::Allocator Get()
	{
		static UmpirePool instance; // call constructor once 
		auto &urm = umpire::ResourceManager::getInstance();
		return urm.getAllocator(name); // get the allocator in name 
	}
private:
	// create allocator 
	UmpirePool()
	{
		auto &urm = umpire::ResourceManager::getInstance();
		auto allocator = urm.getAllocator("HOST");
		urm.makeAllocator<strategy>(name, allocator);				
	}
	// on end of program, warn if data has not all been deallocated 
	~UmpirePool() {
		auto allocator = Get();
		const auto size = allocator.getCurrentSize();
		if (size > 0) 
		{
			MFEM_WARNING(name << " has " << size << " bytes still allocated"); 
		}
	}
};

// name for DynamicPoolList strategy 
constexpr char host_dynamic_pool_name[] = "host_dynamic_pool";
// name for QuickPool strategy 
constexpr char host_quick_pool_name[] = "host_quick_pool";

}

// alias for creating a DynamicPoolList pool 
using DynamicPoolAllocator = internal::UmpirePool<umpire::strategy::DynamicPoolList,internal::host_dynamic_pool_name>;
// alias for creating a QuickPool pool 
using QuickPoolAllocator = internal::UmpirePool<umpire::strategy::QuickPool,internal::host_quick_pool_name>;

// wrap MFEM vector to use an umpire allocator 
// creates a non-owning mfem::Vector 
// destructor ensures memory is deallocated 
class UmpireVector : public mfem::Vector
{
private:
	umpire::Allocator allocator;
public:
	UmpireVector(int size, umpire::Allocator allocator)
		: allocator(allocator), 
		  mfem::Vector(static_cast<double*>(allocator.allocate(size*sizeof(double))), size)
	{ }
	~UmpireVector()
	{
		allocator.deallocate(static_cast<double*>(data));
	}
};

template<typename T> 
class UmpireArray : public mfem::Array<T>
{
private:
	umpire::Allocator allocator;
public:
	UmpireArray(int size, umpire::Allocator allocator)
		: allocator(allocator), 
		  mfem::Array<T>(static_cast<T*>(allocator.allocate(size*sizeof(T))), size)
	{ }
	~UmpireArray()
	{
		allocator.deallocate(static_cast<T*>(mfem::Array<T>::GetMemory()));
	}
};

// generic framework for std::unique_ptr 
// to umpire-managed resources 
namespace internal
{

struct UmpireDeleter
{
	umpire::Allocator allocator;
	template<typename T>
	void operator()(T *ptr) 
	{
		allocator.deallocate(ptr);
	}
};

}

template<typename T> 
using UmpireUniquePtr = std::unique_ptr<T,internal::UmpireDeleter>;

template<typename T> 
inline UmpireUniquePtr<T> CreateUmpireUniquePtr(int size, umpire::Allocator allocator)
{
	auto *ptr = static_cast<T*>(allocator.allocate(size * sizeof(T)));
	return UmpireUniquePtr<T>(ptr, internal::UmpireDeleter(allocator));
}