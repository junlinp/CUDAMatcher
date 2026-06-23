#ifndef MATCH_H_
#define MATCH_H_

#include <array>
#include <vector>

using Descriptor = std::array<float, 128>;

struct KernelBenchmarkResult {
	bool success;
	float best_ms;
	float avg_ms;
	int mismatch_count;
};

struct MatchV10Context;

bool MatchV1(const std::vector< Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result);
bool MatchV2(const std::vector< Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result);
bool MatchV3(const std::vector< Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result);
bool MatchV4(const std::vector< Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result);
bool MatchV5(const std::vector< Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result);
bool MatchV5a(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result);
bool MatchV5b(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result);
bool MatchV5c(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result);
bool MatchV6(const std::vector< Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result);
bool MatchV7(const std::vector< Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result);
bool MatchV8(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result);
bool MatchV9(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result);
bool MatchV10(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result);
bool Match(const std::vector< Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result);
bool CreateMatchV10Context(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs, MatchV10Context** context);
bool RunMatchV10Context(MatchV10Context* context, std::vector<std::pair<int, int>>& match_result);
void DestroyMatchV10Context(MatchV10Context* context);
bool BenchmarkKernelV1(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs,const std::vector<int>& expected_match, int warmup_runs, int measured_runs, KernelBenchmarkResult& result);
bool BenchmarkKernelV2(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs,const std::vector<int>& expected_match, int warmup_runs, int measured_runs, KernelBenchmarkResult& result);
bool BenchmarkKernelV3(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs,const std::vector<int>& expected_match, int warmup_runs, int measured_runs, KernelBenchmarkResult& result);
bool BenchmarkKernelV4(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs,const std::vector<int>& expected_match, int warmup_runs, int measured_runs, KernelBenchmarkResult& result);
bool BenchmarkKernelV5(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs,const std::vector<int>& expected_match, int warmup_runs, int measured_runs, KernelBenchmarkResult& result);
bool BenchmarkKernelV5a(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs,const std::vector<int>& expected_match, int warmup_runs, int measured_runs, KernelBenchmarkResult& result);
bool BenchmarkKernelV5b(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs,const std::vector<int>& expected_match, int warmup_runs, int measured_runs, KernelBenchmarkResult& result);
bool BenchmarkKernelV5c(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs,const std::vector<int>& expected_match, int warmup_runs, int measured_runs, KernelBenchmarkResult& result);
bool BenchmarkKernelV6(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs,const std::vector<int>& expected_match, int warmup_runs, int measured_runs, KernelBenchmarkResult& result);
bool BenchmarkKernelV7(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs,const std::vector<int>& expected_match, int warmup_runs, int measured_runs, KernelBenchmarkResult& result);
bool BenchmarkKernelV8(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs,const std::vector<int>& expected_match, int warmup_runs, int measured_runs, KernelBenchmarkResult& result);
bool BenchmarkKernelV9(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs,const std::vector<int>& expected_match, int warmup_runs, int measured_runs, KernelBenchmarkResult& result);
bool BenchmarkKernelV10(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs,const std::vector<int>& expected_match, int warmup_runs, int measured_runs, KernelBenchmarkResult& result);

#endif  // MATCH_H_
