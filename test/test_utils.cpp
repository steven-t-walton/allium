#include "gtest/gtest.h"
#include "utils.hpp"

class TableTest : public ::testing::Test
{
protected:
	utils::InterpolatedTable1D *table;
	void SetUp() override {
		mfem::Vector x(4), y(4);
		std::iota(x.begin(), x.end(), 0.0);
		std::iota(y.begin(), y.end(), 10.0);
		table = new utils::InterpolatedTable1D(x,y);
	}
	void TearDown() override {
		delete table;
	}
};

TEST_F(TableTest, Min)
{
	EXPECT_DOUBLE_EQ(table->Eval(-1.0), 10.0);	
}

TEST_F(TableTest, Max)
{
	EXPECT_DOUBLE_EQ(table->Eval(10.0), 13.0);		
}

TEST_F(TableTest, Eval)
{
	EXPECT_DOUBLE_EQ(table->Eval(1.25), 11.0*.75 + 12.0*.25);
	EXPECT_DOUBLE_EQ(table->Eval(0.5), 10.0*0.5 + 11.0*0.5);
	EXPECT_DOUBLE_EQ(table->Eval(2.75), 12.0*0.25 + 13.0*0.75);
}

class SemiLogTableTest : public ::testing::Test
{
protected:
	utils::InterpolatedTable1D *table;
	void SetUp() override {
		mfem::Vector x(4), y(4);
		std::iota(x.begin(), x.end(), 0.0);
		std::iota(y.begin(), y.end(), 10.0);
		std::transform(x.begin(), x.end(), x.begin(), [](const double v) { return std::pow(10.0,v); });
		table = new utils::InterpolatedTable1D(x,y);
		table->UseLogX();
	}
	void TearDown() override {
		delete table;
	}
};

TEST_F(SemiLogTableTest, Min) {
	EXPECT_DOUBLE_EQ(table->Eval(0.0), 10.0);
}

TEST_F(SemiLogTableTest, Max) {
	EXPECT_DOUBLE_EQ(table->Eval(1e5), 13.0);
}

TEST_F(SemiLogTableTest, Eval)
{
	EXPECT_DOUBLE_EQ(table->Eval(std::pow(10.0,1.25)), 11.0*.75 + 12.0*.25);
	EXPECT_DOUBLE_EQ(table->Eval(std::pow(10.0, 0.5)), 10.0*0.5 + 11.0*0.5);
	EXPECT_DOUBLE_EQ(table->Eval(std::pow(10.0, 2.75)), 12.0*0.25 + 13.0*0.75);
}

class LogLogTableTest : public ::testing::Test
{
protected:
	utils::InterpolatedTable1D *table;
	void SetUp() override {
		mfem::Vector x(4), y(4);
		std::iota(x.begin(), x.end(), 0.0);
		std::iota(y.begin(), y.end(), 10.0);
		std::transform(x.begin(), x.end(), x.begin(), [](const double v) { return std::pow(10.0,v); });
		std::transform(y.begin(), y.end(), y.begin(), [](const double v) { return std::exp(v); });
		table = new utils::InterpolatedTable1D(x,y);
		table->UseLogX();
		table->UseLogY();
	}
	void TearDown() override {
		delete table;
	}
};

TEST_F(LogLogTableTest, Min) {
	EXPECT_DOUBLE_EQ(table->Eval(0.0), std::exp(10.0));
}

TEST_F(LogLogTableTest, Max) {
	EXPECT_DOUBLE_EQ(table->Eval(1e5), std::exp(13.0));
}

TEST_F(LogLogTableTest, Eval)
{
	EXPECT_DOUBLE_EQ(table->Eval(std::pow(10.0,1.25)), std::exp(11.0*.75 + 12.0*.25));
	EXPECT_DOUBLE_EQ(table->Eval(std::pow(10.0, 0.5)), std::exp(10.0*0.5 + 11.0*0.5));
	EXPECT_DOUBLE_EQ(table->Eval(std::pow(10.0, 2.75)), std::exp(12.0*0.25 + 13.0*0.75));
}

class PiecewiseConstantTableTest : public ::testing::Test
{
protected:
	utils::InterpolatedTable1D *table;
	void SetUp() override {
		mfem::Vector x(4), y(4);
		std::iota(x.begin(), x.end(), 0.0);
		std::iota(y.begin(), y.end(), 10.0);
		table = new utils::InterpolatedTable1D(x,y);
		table->UsePiecewiseConstant();
	}
	void TearDown() override {
		delete table;
	}
};

TEST_F(PiecewiseConstantTableTest, Eval) 
{
	EXPECT_DOUBLE_EQ(table->Eval(0.75), 10.0);
	EXPECT_DOUBLE_EQ(table->Eval(1.125), 11.0);
	EXPECT_DOUBLE_EQ(table->Eval(2.95), 12.0);
}