#include "match.h"

#include <algorithm>
#include <cstdio>

#include "cuda_runtime.h"

__global__ void ComputeDistanceMatrixV1(float* pts1, float* pts2, float* distance_matrix, int WIDTH) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int p1_base = blockIdx.x * blockDim.x;
	__shared__ float4 buffer_pts1[32 * 32];
	__shared__ float4 buffer_pts2[32 * 32];
	__shared__ float score[32 * 32];
	__shared__ float rotation_score[32 * 32];

	for (int i = 0; i < 4; i++) {
		buffer_pts1[(ty * 4 + i) * 32 + tx] = ((float4*)pts1)[(p1_base + ty * 4 + i) * 32 + tx];
	}
	__syncthreads();

	for (int p2 = 0; p2 < WIDTH; p2 += 32) {
		for (int i = 0; i < 4; i++) {
			buffer_pts2[(ty * 4 + i) * 32 + tx] = ((float4*)pts2)[(p2 + ty * 4 + i) * 32 + tx];
		}
		__syncthreads();
		if (ty < 4) {
			float ss[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
			for (int i = 0; i < 32; i++) {
				float4 v1[2];
				for (int dx = 0; dx < 2; dx++) {
					v1[dx] = buffer_pts1[(16 * dx + tx) % 32 * 32 + (i + tx) % 32];
				}

				for (int dy = 0; dy < 4; dy++) {
					float4 pt2 = buffer_pts2[(4 * (2 * ty + tx / 16) + dy) * 32 + (i + tx) % 32];
					float a = pt2.x - v1[0].x;
					ss[dy] += a * a;
					a = pt2.y - v1[0].y;
					ss[dy] += a * a;
					a = pt2.z - v1[0].z;
					ss[dy] += a * a;
					a = pt2.w - v1[0].w;
					ss[dy] += a * a;

					a = pt2.x - v1[1].x;
					ss[4 + dy] += a * a;
					a = pt2.y - v1[1].y;
					ss[4 + dy] += a * a;
					a = pt2.z - v1[1].z;
					ss[4 + dy] += a * a;
					a = pt2.w - v1[1].w;
					ss[4 + dy] += a * a;
				}
			}
			for (int dy = 0; dy < 4; dy++) {
				for (int dx = 0; dx < 2; dx++) {
					int row = (tx + 16 * dx) % 32;
					int col = (4 * (2 * ty + tx / 16) + dy);
					score[row + 32 * col] = ss[dx * 4 + dy];
				}
			}
		}
		__syncthreads();
		for (int i = 0; i < 4; i++) {
			rotation_score[(tx + (4 * ty + i)) % 32 + 32 * (4 * ty + i)] = score[tx + 32 * (4 * ty + i)];
		}
		__syncthreads();
		for (int i = 0; i < 4; i++) {
			distance_matrix[(p1_base + 4 * ty + i) * WIDTH + p2 + tx] =
			    rotation_score[tx * 32 + (ty * 4 + i + tx) % 32];
		}
	}
}

#define BATCH_NUM_ 256
__global__ void CloseElementV1(float* distance_matrix, int WIDTH, float* score, int* index) {
	int p1 = blockDim.x * blockIdx.x + threadIdx.x;
	int p1_base = blockDim.x * blockIdx.x;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int index_ = -1;
	float minimum_score = 1e10f;

	__shared__ float buffer_distance[32 * BATCH_NUM_];
	__shared__ int buffer_index[32 * 32];
	float d = 0.0f;
	for (int i = 0; i < WIDTH; i += BATCH_NUM_) {
		for (int batch_index = 0; batch_index < BATCH_NUM_ / 32; batch_index++) {
			buffer_distance[ty * BATCH_NUM_ + (tx + ty) % 32 + batch_index * 32] =
			    distance_matrix[(p1_base + ty) * WIDTH + i + tx + batch_index * 32];
		}
		__syncthreads();
		for (int batch_index = 0; batch_index < BATCH_NUM_ / 32; batch_index++) {
			d = buffer_distance[tx * BATCH_NUM_ + (tx + ty) % 32 + batch_index * 32];
			if (d < minimum_score) {
				minimum_score = d;
				index_ = i + batch_index * 32 + ty;
			}
		}
		__syncthreads();
	}
	__syncthreads();
	buffer_distance[tx * 32 + (ty + tx) % 32] = minimum_score;
	buffer_index[tx * 32 + (ty + tx) % 32] = index_;
	__syncthreads();
	if (ty == 0) {
		for (int i = 0; i < 32; i++) {
			float dd = buffer_distance[tx * 32 + (i + tx) % 32];
			if (dd < minimum_score) {
				minimum_score = dd;
				index_ = buffer_index[tx * 32 + (i + tx) % 32];
			}
		}
		score[p1] = minimum_score;
		index[p1] = index_;
	}
}

#define CUDAErrorCheck(x) ErrorCheckImp((x), __FILE__, __LINE__)

bool ErrorCheckImp(cudaError_t code, const char* file_name, const int line) {
	if (code != cudaSuccess) {
		fprintf(stderr, "ERROR Checked : %s in %s %d\n", cudaGetErrorString(code), file_name, line);
		return false;
	}
	return true;
}

void CopyDescriptorToPinnedMemory(const std::vector<Descriptor>& descriptors, float* host_memory) {
	size_t offset = 0;
	for (const Descriptor& descriptor : descriptors) {
		std::copy(descriptor.begin(), descriptor.end(), host_memory + offset);
		offset += descriptor.size();
	}
}

bool MatchV1(const std::vector<Descriptor>& lhs,
             const std::vector<Descriptor>& rhs,
             std::vector<std::pair<int, int>>& match_result) {
	size_t size = lhs.size();
	if (size == 0 || size != rhs.size() || size % 256 != 0) {
		fprintf(stderr, "ERROR Checked : invalid descriptor sizes\n");
		return false;
	}
	match_result.clear();

	float* lhs_descriptor_host = nullptr;
	float* rhs_descriptor_host = nullptr;
	float* lhs_descriptor_device = nullptr;
	float* rhs_descriptor_device = nullptr;
	float* distance_matrix = nullptr;
	float* device_score = nullptr;
	int* host_index = nullptr;
	int* device_index = nullptr;
	dim3 distance_thread(32, 8);
	dim3 distance_block(size / 32, 1);
	dim3 close_thread(32, 32);
	dim3 close_block(size / 32, 1);
	bool success = false;

	if (!CUDAErrorCheck(cudaMallocHost(&lhs_descriptor_host, sizeof(float) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMallocHost(&rhs_descriptor_host, sizeof(float) * size * 128))) goto cleanup;
	CopyDescriptorToPinnedMemory(lhs, lhs_descriptor_host);
	CopyDescriptorToPinnedMemory(rhs, rhs_descriptor_host);

	if (!CUDAErrorCheck(cudaMalloc(&lhs_descriptor_device, sizeof(float) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&rhs_descriptor_device, sizeof(float) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(lhs_descriptor_device, lhs_descriptor_host, sizeof(float) * size * 128, cudaMemcpyHostToDevice))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(rhs_descriptor_device, rhs_descriptor_host, sizeof(float) * size * 128, cudaMemcpyHostToDevice))) goto cleanup;

	if (!CUDAErrorCheck(cudaMalloc(&distance_matrix, sizeof(float) * size * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMallocHost(&host_index, sizeof(int) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&device_score, sizeof(float) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&device_index, sizeof(int) * size))) goto cleanup;

	ComputeDistanceMatrixV1<<<distance_block, distance_thread>>>(lhs_descriptor_device, rhs_descriptor_device, distance_matrix, size);
	if (!CUDAErrorCheck(cudaGetLastError())) goto cleanup;
	CloseElementV1<<<close_block, close_thread>>>(distance_matrix, size, device_score, device_index);
	if (!CUDAErrorCheck(cudaGetLastError())) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(host_index, device_index, sizeof(int) * size, cudaMemcpyDeviceToHost))) goto cleanup;

	for (size_t i = 0; i < size; i++) {
		match_result.push_back(std::pair<int, int>(static_cast<int>(i), host_index[i]));
	}
	success = true;

cleanup:
	if (lhs_descriptor_host) cudaFreeHost(lhs_descriptor_host);
	if (rhs_descriptor_host) cudaFreeHost(rhs_descriptor_host);
	if (lhs_descriptor_device) cudaFree(lhs_descriptor_device);
	if (rhs_descriptor_device) cudaFree(rhs_descriptor_device);
	if (distance_matrix) cudaFree(distance_matrix);
	if (device_score) cudaFree(device_score);
	if (device_index) cudaFree(device_index);
	if (host_index) cudaFreeHost(host_index);
	if (!success) match_result.clear();
	return success;
}

bool BenchmarkKernelV1(const std::vector<Descriptor>& lhs,
                       const std::vector<Descriptor>& rhs,
                       const std::vector<int>& expected_match,
                       int warmup_runs,
                       int measured_runs,
                       KernelBenchmarkResult& result) {
	size_t size = lhs.size();
	result.success = false;
	result.best_ms = 0.0f;
	result.avg_ms = 0.0f;
	result.mismatch_count = static_cast<int>(expected_match.size());
	if (size == 0 || size != rhs.size() || size != expected_match.size() || size % 256 != 0 || warmup_runs < 0 || measured_runs <= 0) {
		fprintf(stderr, "ERROR Checked : invalid kernel benchmark input\n");
		return false;
	}

	float* lhs_descriptor_host = nullptr;
	float* rhs_descriptor_host = nullptr;
	float* lhs_descriptor_device = nullptr;
	float* rhs_descriptor_device = nullptr;
	float* distance_matrix = nullptr;
	float* device_score = nullptr;
	int* host_index = nullptr;
	int* device_index = nullptr;
	cudaEvent_t start = nullptr;
	cudaEvent_t stop = nullptr;
	dim3 distance_thread(32, 8);
	dim3 distance_block(size / 32, 1);
	dim3 close_thread(32, 32);
	dim3 close_block(size / 32, 1);
	bool success = false;
	float best_ms = 1e30f;
	float sum_ms = 0.0f;

	if (!CUDAErrorCheck(cudaMallocHost(&lhs_descriptor_host, sizeof(float) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMallocHost(&rhs_descriptor_host, sizeof(float) * size * 128))) goto cleanup;
	CopyDescriptorToPinnedMemory(lhs, lhs_descriptor_host);
	CopyDescriptorToPinnedMemory(rhs, rhs_descriptor_host);
	if (!CUDAErrorCheck(cudaMalloc(&lhs_descriptor_device, sizeof(float) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&rhs_descriptor_device, sizeof(float) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&distance_matrix, sizeof(float) * size * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&device_score, sizeof(float) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&device_index, sizeof(int) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMallocHost(&host_index, sizeof(int) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(lhs_descriptor_device, lhs_descriptor_host, sizeof(float) * size * 128, cudaMemcpyHostToDevice))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(rhs_descriptor_device, rhs_descriptor_host, sizeof(float) * size * 128, cudaMemcpyHostToDevice))) goto cleanup;

	for (int i = 0; i < warmup_runs; i++) {
		ComputeDistanceMatrixV1<<<distance_block, distance_thread>>>(lhs_descriptor_device, rhs_descriptor_device, distance_matrix, size);
		CloseElementV1<<<close_block, close_thread>>>(distance_matrix, size, device_score, device_index);
	}
	if (!CUDAErrorCheck(cudaGetLastError())) goto cleanup;
	if (!CUDAErrorCheck(cudaDeviceSynchronize())) goto cleanup;

	if (!CUDAErrorCheck(cudaEventCreate(&start))) goto cleanup;
	if (!CUDAErrorCheck(cudaEventCreate(&stop))) goto cleanup;
	for (int i = 0; i < measured_runs; i++) {
		float elapsed_ms = 0.0f;
		if (!CUDAErrorCheck(cudaEventRecord(start, 0))) goto cleanup;
		ComputeDistanceMatrixV1<<<distance_block, distance_thread>>>(lhs_descriptor_device, rhs_descriptor_device, distance_matrix, size);
		CloseElementV1<<<close_block, close_thread>>>(distance_matrix, size, device_score, device_index);
		if (!CUDAErrorCheck(cudaEventRecord(stop, 0))) goto cleanup;
		if (!CUDAErrorCheck(cudaEventSynchronize(stop))) goto cleanup;
		if (!CUDAErrorCheck(cudaGetLastError())) goto cleanup;
		if (!CUDAErrorCheck(cudaEventElapsedTime(&elapsed_ms, start, stop))) goto cleanup;
		sum_ms += elapsed_ms;
		if (elapsed_ms < best_ms) {
			best_ms = elapsed_ms;
		}
	}

	if (!CUDAErrorCheck(cudaMemcpy(host_index, device_index, sizeof(int) * size, cudaMemcpyDeviceToHost))) goto cleanup;
	result.mismatch_count = 0;
	for (size_t i = 0; i < size; i++) {
		if (host_index[i] != expected_match[i]) {
			result.mismatch_count++;
		}
	}
	result.best_ms = best_ms;
	result.avg_ms = sum_ms / static_cast<float>(measured_runs);
	result.success = result.mismatch_count == 0;
	success = result.success;

cleanup:
	if (start) cudaEventDestroy(start);
	if (stop) cudaEventDestroy(stop);
	if (lhs_descriptor_host) cudaFreeHost(lhs_descriptor_host);
	if (rhs_descriptor_host) cudaFreeHost(rhs_descriptor_host);
	if (lhs_descriptor_device) cudaFree(lhs_descriptor_device);
	if (rhs_descriptor_device) cudaFree(rhs_descriptor_device);
	if (distance_matrix) cudaFree(distance_matrix);
	if (device_score) cudaFree(device_score);
	if (device_index) cudaFree(device_index);
	if (host_index) cudaFreeHost(host_index);
	return success;
}

bool Match(const std::vector<Descriptor>& lhs,
           const std::vector<Descriptor>& rhs,
           std::vector<std::pair<int, int>>& match_result) {
	return MatchV1(lhs, rhs, match_result);
}
