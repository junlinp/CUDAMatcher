#ifndef MATCH_H_
#define MATCH_H_

#include <array>
#include <utility>
#include <vector>

using Descriptor = std::array<float, 128>;

struct KernelBenchmarkResult {
	bool success;
	float best_ms;
	float avg_ms;
	int mismatch_count;
};

bool MatchV1(const std::vector<Descriptor>& lhs,
             const std::vector<Descriptor>& rhs,
             std::vector<std::pair<int, int>>& match_result);
bool Match(const std::vector<Descriptor>& lhs,
           const std::vector<Descriptor>& rhs,
           std::vector<std::pair<int, int>>& match_result);
bool BenchmarkKernelV1(const std::vector<Descriptor>& lhs,
                       const std::vector<Descriptor>& rhs,
                       const std::vector<int>& expected_match,
                       int warmup_runs,
                       int measured_runs,
                       KernelBenchmarkResult& result);

#endif  // MATCH_H_
