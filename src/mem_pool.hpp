#pragma once 

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPoolList.hpp"
#include "umpire/strategy/QuickPool.hpp"

class DynamicPoolAllocator
{
public:
	static umpire::Allocator Get()
	{
		static DynamicPoolAllocator instance;
		auto &urm = umpire::ResourceManager::getInstance();
		return urm.getAllocator(name);		
	}
private:
	DynamicPoolAllocator() {
		auto &urm = umpire::ResourceManager::getInstance();
		auto allocator = urm.getAllocator("HOST");
		urm.makeAllocator<umpire::strategy::DynamicPoolList>(name, allocator, 1024*sizeof(double));		
	}
	static constexpr char name[] = "host_dynamic_pool";
};

class QuickPoolAllocator
{
public:
	static umpire::Allocator Get()
	{
		static QuickPoolAllocator instance;
		auto &urm = umpire::ResourceManager::getInstance();
		return urm.getAllocator(name);		
	}
private:
	QuickPoolAllocator() {
		auto &urm = umpire::ResourceManager::getInstance();
		auto allocator = urm.getAllocator("HOST");
		urm.makeAllocator<umpire::strategy::QuickPool>(name, allocator, 1024*sizeof(double));		
	}
	static constexpr char name[] = "host_quick_pool";
};
