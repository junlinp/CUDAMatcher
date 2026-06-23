#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "match.h"

struct BenchmarkResult {
	bool success;
	double wall_best_ms;
	double wall_avg_ms;
	float device_best_ms;
	float device_avg_ms;
	int mismatch_count;
};

double EstimatedFlops(size_t descriptor_num) {
	double n = static_cast<double>(descriptor_num);
	double k = 128.0;
	return n * n * (3.0 * k);
}

double EstimatedLogicalBytes(size_t descriptor_num) {
	double n = static_cast<double>(descriptor_num);
	double k = 128.0;
	double descriptor_scan = (n / 32.0) * (32.0 * k * 4.0 + n * k * 4.0);
	double matrix_write_read = 2.0 * n * n * 4.0;
	return descriptor_scan + matrix_write_read;
}

void PrintMetrics(const std::string& name, const BenchmarkResult& result, size_t descriptor_num) {
	if (!result.success || result.wall_best_ms <= 0.0) {
		return;
	}
	double seconds = result.wall_best_ms / 1000.0;
	double gflops = EstimatedFlops(descriptor_num) / seconds / 1e9;
	double logical_gbytes = EstimatedLogicalBytes(descriptor_num) / seconds / 1e9;
	std::cout << name << " wall best : " << std::fixed << std::setprecision(3) << result.wall_best_ms << " ms" << std::endl;
	std::cout << name << " wall avg : " << std::fixed << std::setprecision(3) << result.wall_avg_ms << " ms" << std::endl;
	std::cout << name << " wall-best algorithmic compute : " << std::fixed << std::setprecision(2) << gflops << " GFLOP/s" << std::endl;
	std::cout << name << " wall-best estimated logical bytes : " << std::fixed << std::setprecision(2) << logical_gbytes << " GB/s" << std::endl;
	if (result.device_best_ms > 0.0f) {
		double device_seconds = static_cast<double>(result.device_best_ms) / 1000.0;
		double device_gflops = EstimatedFlops(descriptor_num) / device_seconds / 1e9;
		double device_logical_gbytes = EstimatedLogicalBytes(descriptor_num) / device_seconds / 1e9;
		std::cout << name << " device best : " << std::fixed << std::setprecision(3) << result.device_best_ms << " ms" << std::endl;
		std::cout << name << " device avg : " << std::fixed << std::setprecision(3) << result.device_avg_ms << " ms" << std::endl;
		std::cout << name << " device-best algorithmic compute : " << std::fixed << std::setprecision(2) << device_gflops << " GFLOP/s" << std::endl;
		std::cout << name << " device-best estimated logical bytes : " << std::fixed << std::setprecision(2) << device_logical_gbytes << " GB/s" << std::endl;
	}
}

void PrintKernelMetrics(const KernelBenchmarkResult& result, size_t descriptor_num) {
	std::cout << "v1 kernel-only mismatch count : " << result.mismatch_count << std::endl;
	if (!result.success || result.best_ms <= 0.0f) {
		std::cout << "v1 kernel-only benchmark failed" << std::endl;
		return;
	}
	double seconds = static_cast<double>(result.best_ms) / 1000.0;
	double gflops = EstimatedFlops(descriptor_num) / seconds / 1e9;
	double logical_gbytes = EstimatedLogicalBytes(descriptor_num) / seconds / 1e9;
	std::cout << "v1 kernel-only best : " << std::fixed << std::setprecision(3) << result.best_ms << " ms" << std::endl;
	std::cout << "v1 kernel-only avg : " << std::fixed << std::setprecision(3) << result.avg_ms << " ms" << std::endl;
	std::cout << "v1 kernel-only algorithmic compute : " << std::fixed << std::setprecision(2) << gflops << " GFLOP/s" << std::endl;
	std::cout << "v1 kernel-only estimated logical bytes : " << std::fixed << std::setprecision(2) << logical_gbytes << " GB/s" << std::endl;
}

BenchmarkResult RunBenchmark(const std::vector<Descriptor>& lhs,
                             const std::vector<Descriptor>& rhs,
                             const std::vector<int>& expected_match,
                             size_t descriptor_num) {
	const int warmup_runs = 1;
	const int measured_runs = 5;
	bool success = true;
	int mismatch_count = 0;
	double wall_sum_ms = 0.0;
	double wall_best_ms = std::numeric_limits<double>::max();
	float device_sum_ms = 0.0f;
	float device_best_ms = std::numeric_limits<float>::max();

	for (int run = 0; run < warmup_runs + measured_runs; run++) {
		std::vector<std::pair<int, int>> match_result;
		cudaEvent_t device_start = nullptr;
		cudaEvent_t device_stop = nullptr;
		float device_elapsed_ms = 0.0f;
		cudaEventCreate(&device_start);
		cudaEventCreate(&device_stop);
		cudaEventRecord(device_start, 0);
		auto start = std::chrono::high_resolution_clock::now();
		bool run_success = MatchV1(lhs, rhs, match_result);
		auto end = std::chrono::high_resolution_clock::now();
		cudaEventRecord(device_stop, 0);
		cudaEventSynchronize(device_stop);
		cudaEventElapsedTime(&device_elapsed_ms, device_start, device_stop);
		cudaEventDestroy(device_start);
		cudaEventDestroy(device_stop);

		double wall_elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
		int run_mismatch_count = 0;
		if (run_success) {
			for (std::pair<int, int> p : match_result) {
				if (p.second != expected_match[p.first]) {
					run_mismatch_count++;
				}
			}
		} else {
			run_mismatch_count = static_cast<int>(expected_match.size());
		}

		success = success && run_success;
		if (run_mismatch_count > mismatch_count) {
			mismatch_count = run_mismatch_count;
		}

		if (run >= warmup_runs) {
			wall_sum_ms += wall_elapsed_ms;
			device_sum_ms += device_elapsed_ms;
			if (wall_elapsed_ms < wall_best_ms) {
				wall_best_ms = wall_elapsed_ms;
			}
			if (device_elapsed_ms < device_best_ms) {
				device_best_ms = device_elapsed_ms;
			}
		}
	}

	BenchmarkResult result;
	result.success = success;
	result.wall_best_ms = wall_best_ms;
	result.wall_avg_ms = wall_sum_ms / static_cast<double>(measured_runs);
	result.device_best_ms = device_best_ms;
	result.device_avg_ms = device_sum_ms / static_cast<float>(measured_runs);
	result.mismatch_count = mismatch_count;

	std::cout << "v1 measured runs : " << measured_runs << " + " << warmup_runs << " warmup" << std::endl;
	std::cout << "v1 mismatch count : " << mismatch_count << std::endl;
	if (!success) {
		std::cout << "v1 match failed" << std::endl;
	}
	PrintMetrics("v1", result, descriptor_num);
	return result;
}

void MATCH() {
	const size_t descriptor_num = 1024 * 16;

	std::vector<Descriptor> lhs(descriptor_num), rhs(descriptor_num);
	std::vector<int> shuffle(descriptor_num);
	std::default_random_engine engine;
	std::uniform_real_distribution<float> distribute(0, 255);
	for (size_t i = 0; i < descriptor_num; i++) {
		Descriptor descriptor;
		for (int k = 0; k < 128; k++) {
			descriptor[k] = distribute(engine);
		}
		shuffle[i] = static_cast<int>(descriptor_num - 1 - i);
		lhs[i] = descriptor;
	}

	std::shuffle(shuffle.begin(), shuffle.end(), engine);
	for (size_t i = 0; i < descriptor_num; i++) {
		rhs[i] = lhs[shuffle[i]];
	}
	std::vector<int> expected_match(descriptor_num);
	for (size_t i = 0; i < descriptor_num; i++) {
		expected_match[shuffle[i]] = static_cast<int>(i);
	}

	RunBenchmark(lhs, rhs, expected_match, descriptor_num);

	const int warmup_runs = 5;
	const int measured_runs = 20;
	KernelBenchmarkResult kernel_result;
	std::cout << "kernel-only measured runs : " << measured_runs << " + " << warmup_runs << " warmup" << std::endl;
	BenchmarkKernelV1(lhs, rhs, expected_match, warmup_runs, measured_runs, kernel_result);
	PrintKernelMetrics(kernel_result, descriptor_num);
}

int main(int argc, char** argv) {
	MATCH();
	return 0;
}
