
#include <vector>
#include <array>
#include <chrono>
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <string>
#include "match.h"

struct BenchmarkResult {
    bool success;
    long long elapsed_ms;
    int mismatch_count;
};

double EstimatedFlops(const std::string& name, size_t descriptor_num) {
    double n = static_cast<double>(descriptor_num);
    double k = 128.0;
    if (name == "v9") {
        return n * n * (2.0 * k + 3.0);
    }
    return n * n * (3.0 * k);
}

double EstimatedMemoryBytes(const std::string& name, size_t descriptor_num) {
    double n = static_cast<double>(descriptor_num);
    double k = 128.0;
    if (name == "v1") {
        double descriptor_scan = (n / 32.0) * (32.0 * k * 4.0 + n * k * 4.0);
        double matrix_write_read = 2.0 * n * n * 4.0;
        return descriptor_scan + matrix_write_read;
    }
    if (name == "v4") {
        return (n / 8.0) * (8.0 * k * 4.0 + n * k * 4.0);
    }
    if (name == "v8") {
        return (n / 32.0) * (32.0 * k * 2.0 + n * k * 2.0);
    }
    if (name == "v9") {
        double descriptor_scan = (n / 16.0) * (16.0 * k * 2.0 + n * k * 2.0);
        double norm_scan = (n / 16.0) * (16.0 * 4.0 + n * 4.0);
        return descriptor_scan + norm_scan;
    }
    return (n / 32.0) * (32.0 * k * 4.0 + n * k * 4.0);
}

void PrintMetrics(const std::string& name, const BenchmarkResult& result, size_t descriptor_num) {
    if (!result.success || result.elapsed_ms <= 0) {
        return;
    }
    double seconds = static_cast<double>(result.elapsed_ms) / 1000.0;
    double gflops = EstimatedFlops(name, descriptor_num) / seconds / 1e9;
    double bandwidth = EstimatedMemoryBytes(name, descriptor_num) / seconds / 1e9;
    std::cout << name << " estimated compute : " << std::fixed << std::setprecision(2) << gflops << " GFLOP/s" << std::endl;
    std::cout << name << " estimated bandwidth : " << std::fixed << std::setprecision(2) << bandwidth << " GB/s" << std::endl;
}

BenchmarkResult RunBenchmark(const std::string& name,
                             bool (*matcher)(const std::vector<Descriptor>&,
                                             const std::vector<Descriptor>&,
                                             std::vector<std::pair<int, int>>&),
                             const std::vector<Descriptor>& lhs,
                             const std::vector<Descriptor>& rhs,
                             const std::vector<int>& expected_match,
                             size_t descriptor_num) {
    std::vector<std::pair<int, int>> match_result;
    auto start = std::chrono::high_resolution_clock::now();
    bool success = matcher(lhs, rhs, match_result);
    auto end = std::chrono::high_resolution_clock::now();
    long long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    int mismatch_count = 0;
    if (success) {
        for (std::pair<int, int> p : match_result) {
            if (p.second != expected_match[p.first]) {
                mismatch_count++;
            }
        }
    } else {
        mismatch_count = static_cast<int>(expected_match.size());
    }

    std::cout << name << " time elapsed : " << elapsed_ms << " ms" << std::endl;
    std::cout << name << " mismatch count : " << mismatch_count << std::endl;
    if (!success) {
        std::cout << name << " match failed" << std::endl;
    }

    BenchmarkResult result;
    result.success = success;
    result.elapsed_ms = elapsed_ms;
    result.mismatch_count = mismatch_count;
    PrintMetrics(name, result, descriptor_num);
    return result;
}

void MATCH() {
    const size_t descriptor_num = 1024 * 16;

    std::vector<Descriptor> lhs(descriptor_num), rhs(descriptor_num);
    std::vector<int> shuffle(descriptor_num);
    std::default_random_engine engine;
    std::uniform_real_distribution<float> distribute(0, 255);
    for (int i = 0; i < descriptor_num; i++) {
        Descriptor descriptor;
        for (int k  = 0; k < 128; k++) {
            descriptor[k] = distribute(engine);
        }
        shuffle[i] = descriptor_num - 1 - i;
        lhs[i] = descriptor;
    }

    std::random_shuffle(shuffle.begin(), shuffle.end());
    for (int i = 0; i < descriptor_num; i++) {
        rhs[i] = lhs[shuffle[i]];
    }
    std::vector<int> expected_match(descriptor_num);
    for (int i = 0; i < descriptor_num; i++) {
        expected_match[shuffle[i]] = i;
    }

    BenchmarkResult v1 = RunBenchmark("v1", MatchV1, lhs, rhs, expected_match, descriptor_num);
    BenchmarkResult v2 = RunBenchmark("v2", MatchV2, lhs, rhs, expected_match, descriptor_num);
    BenchmarkResult v3 = RunBenchmark("v3", MatchV3, lhs, rhs, expected_match, descriptor_num);
    BenchmarkResult v4 = RunBenchmark("v4", MatchV4, lhs, rhs, expected_match, descriptor_num);
    BenchmarkResult v5 = RunBenchmark("v5", MatchV5, lhs, rhs, expected_match, descriptor_num);
    BenchmarkResult v5a = RunBenchmark("v5a", MatchV5a, lhs, rhs, expected_match, descriptor_num);
    BenchmarkResult v5b = RunBenchmark("v5b", MatchV5b, lhs, rhs, expected_match, descriptor_num);
    BenchmarkResult v5c = RunBenchmark("v5c", MatchV5c, lhs, rhs, expected_match, descriptor_num);
    BenchmarkResult v6 = RunBenchmark("v6", MatchV6, lhs, rhs, expected_match, descriptor_num);
    BenchmarkResult v7 = RunBenchmark("v7", MatchV7, lhs, rhs, expected_match, descriptor_num);
    BenchmarkResult v8 = RunBenchmark("v8", MatchV8, lhs, rhs, expected_match, descriptor_num);
    BenchmarkResult v9 = RunBenchmark("v9", MatchV9, lhs, rhs, expected_match, descriptor_num);

    if (v1.success && v2.success && v1.mismatch_count == 0 && v2.mismatch_count == 0 && v2.elapsed_ms > 0) {
        double speedup = static_cast<double>(v1.elapsed_ms) / static_cast<double>(v2.elapsed_ms);
        std::cout << "v2 speedup over v1 : " << speedup << "x" << std::endl;
    }
    if (v1.success && v3.success && v1.mismatch_count == 0 && v3.mismatch_count == 0 && v3.elapsed_ms > 0) {
        double speedup = static_cast<double>(v1.elapsed_ms) / static_cast<double>(v3.elapsed_ms);
        std::cout << "v3 speedup over v1 : " << speedup << "x" << std::endl;
    }
    if (v2.success && v3.success && v2.mismatch_count == 0 && v3.mismatch_count == 0 && v3.elapsed_ms > 0) {
        double speedup = static_cast<double>(v2.elapsed_ms) / static_cast<double>(v3.elapsed_ms);
        std::cout << "v3 speedup over v2 : " << speedup << "x" << std::endl;
    }
    if (v1.success && v4.success && v1.mismatch_count == 0 && v4.mismatch_count == 0 && v4.elapsed_ms > 0) {
        double speedup = static_cast<double>(v1.elapsed_ms) / static_cast<double>(v4.elapsed_ms);
        std::cout << "v4 speedup over v1 : " << speedup << "x" << std::endl;
    }
    if (v3.success && v4.success && v3.mismatch_count == 0 && v4.mismatch_count == 0 && v4.elapsed_ms > 0) {
        double speedup = static_cast<double>(v3.elapsed_ms) / static_cast<double>(v4.elapsed_ms);
        std::cout << "v4 speedup over v3 : " << speedup << "x" << std::endl;
    }
    if (v1.success && v5.success && v1.mismatch_count == 0 && v5.mismatch_count == 0 && v5.elapsed_ms > 0) {
        double speedup = static_cast<double>(v1.elapsed_ms) / static_cast<double>(v5.elapsed_ms);
        std::cout << "v5 speedup over v1 : " << speedup << "x" << std::endl;
    }
    if (v4.success && v5.success && v4.mismatch_count == 0 && v5.mismatch_count == 0 && v5.elapsed_ms > 0) {
        double speedup = static_cast<double>(v4.elapsed_ms) / static_cast<double>(v5.elapsed_ms);
        std::cout << "v5 speedup over v4 : " << speedup << "x" << std::endl;
    }
    if (v3.success && v5.success && v3.mismatch_count == 0 && v5.mismatch_count == 0 && v5.elapsed_ms > 0) {
        double speedup = static_cast<double>(v3.elapsed_ms) / static_cast<double>(v5.elapsed_ms);
        std::cout << "v5 speedup over v3 : " << speedup << "x" << std::endl;
    }
    if (v5.success && v5a.success && v5.mismatch_count == 0 && v5a.mismatch_count == 0 && v5a.elapsed_ms > 0) {
        double speedup = static_cast<double>(v5.elapsed_ms) / static_cast<double>(v5a.elapsed_ms);
        std::cout << "v5a speedup over v5 : " << speedup << "x" << std::endl;
    }
    if (v5a.success && v5b.success && v5a.mismatch_count == 0 && v5b.mismatch_count == 0 && v5b.elapsed_ms > 0) {
        double speedup = static_cast<double>(v5a.elapsed_ms) / static_cast<double>(v5b.elapsed_ms);
        std::cout << "v5b speedup over v5a : " << speedup << "x" << std::endl;
    }
    if (v5b.success && v5c.success && v5b.mismatch_count == 0 && v5c.mismatch_count == 0 && v5c.elapsed_ms > 0) {
        double speedup = static_cast<double>(v5b.elapsed_ms) / static_cast<double>(v5c.elapsed_ms);
        std::cout << "v5c speedup over v5b : " << speedup << "x" << std::endl;
    }
    if (v4.success && v6.success && v4.mismatch_count == 0 && v6.mismatch_count == 0 && v6.elapsed_ms > 0) {
        double speedup = static_cast<double>(v4.elapsed_ms) / static_cast<double>(v6.elapsed_ms);
        std::cout << "v6 speedup over v4 : " << speedup << "x" << std::endl;
    }
    if (v5.success && v6.success && v5.mismatch_count == 0 && v6.mismatch_count == 0 && v6.elapsed_ms > 0) {
        double speedup = static_cast<double>(v5.elapsed_ms) / static_cast<double>(v6.elapsed_ms);
        std::cout << "v6 speedup over v5 : " << speedup << "x" << std::endl;
    }
    if (v6.success && v7.success && v6.mismatch_count == 0 && v7.mismatch_count == 0 && v7.elapsed_ms > 0) {
        double speedup = static_cast<double>(v6.elapsed_ms) / static_cast<double>(v7.elapsed_ms);
        std::cout << "v7 speedup over v6 : " << speedup << "x" << std::endl;
    }
    if (v6.success && v8.success && v6.mismatch_count == 0 && v8.mismatch_count == 0 && v8.elapsed_ms > 0) {
        double speedup = static_cast<double>(v6.elapsed_ms) / static_cast<double>(v8.elapsed_ms);
        std::cout << "v8 speedup over v6 : " << speedup << "x" << std::endl;
    }
    if (v8.success && v9.success && v8.mismatch_count == 0 && v9.mismatch_count == 0 && v9.elapsed_ms > 0) {
        double speedup = static_cast<double>(v8.elapsed_ms) / static_cast<double>(v9.elapsed_ms);
        std::cout << "v9 speedup over v8 : " << speedup << "x" << std::endl;
    }
    if (v6.success && v9.success && v6.mismatch_count == 0 && v9.mismatch_count == 0 && v9.elapsed_ms > 0) {
        double speedup = static_cast<double>(v6.elapsed_ms) / static_cast<double>(v9.elapsed_ms);
        std::cout << "v9 speedup over v6 : " << speedup << "x" << std::endl;
    }
}
int main(int argc, char** argv) {
    MATCH();
    return 0;
}
