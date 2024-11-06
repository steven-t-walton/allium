#include "gtest/gtest.h"
#include "mem_pool.hpp"

TEST(Umpire, Vector) {
	auto &rm = umpire::ResourceManager::getInstance();
	auto allocator = rm.getAllocator("HOST");
	{
		UmpireVector v(10000, allocator);
		EXPECT_EQ(allocator.getCurrentSize(), sizeof(double)*10000);
		UmpireVector w(10000, allocator);
		EXPECT_EQ(allocator.getCurrentSize(), sizeof(double)*20000);
	}
	EXPECT_EQ(allocator.getCurrentSize(), 0);
}

TEST(Umpire, UniquePtr) {
	auto &rm = umpire::ResourceManager::getInstance();
	auto allocator = rm.getAllocator("HOST");
	auto v = CreateUmpireUniquePtr<int>(5, allocator);
	EXPECT_EQ(allocator.getCurrentSize(), 20);
	v.reset();
	EXPECT_EQ(allocator.getCurrentSize(), 0);
}