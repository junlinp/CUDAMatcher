#include "match.h"
#include <algorithm>
#include <cstdio>
#include "cuda_runtime.h"

__global__ void ComputeNearestNeighbor(float* pts1, float* pts2, float* score, int* index, int WIDTH) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int p1_base = blockIdx.x * blockDim.x;
	__shared__ float4 buffer_pts1[32 * 32];
	__shared__ float4 buffer_pts2[32 * 32];
	__shared__ float tile_score[32 * 32];
	__shared__ float score_buffer[32 * 32];
	__shared__ float rotation_score[32 * 32];
	__shared__ float best_score[32];
	__shared__ int best_index[32];

	for (int i = 0; i < 4; i++) {
		buffer_pts1[(ty * 4 + i) * 32 + tx] = ((float4*)pts1)[(p1_base + ty * 4 + i) * 32 + tx];
		if (tx == 0) {
			int row = 4 * ty + i;
			best_score[row] = 1e30f;
			best_index[row] = -1;
		}
	}
	__syncthreads();

	for (int p2 = 0; p2 < WIDTH; p2 += 32) {
		for (int i = 0; i < 4; i++) {
			buffer_pts2[(ty * 4 + i) * 32 + tx] = ((float4*)pts2)[(p2 + ty * 4 + i) * 32 + tx];
		}
		__syncthreads();
		if (ty < 4) {
			float ss[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
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
					int col = ( 4 * (2 * ty + tx / 16) + dy);
					score_buffer[row + 32 * col] = ss[dx * 4 + dy];
				}
			}
		}
		__syncthreads();
		for (int i = 0; i < 4; i++) {
			rotation_score[(tx + ( 4 * ty + i)) % 32 + 32 * (4 * ty + i)] = score_buffer[tx + 32 * (4 * ty + i)];
		}
		__syncthreads();
		for (int i = 0; i < 4; i++) {
			int row = 4 * ty + i;
			tile_score[row * 32 + tx] = rotation_score[tx * 32 + (row + tx) % 32];
		}
		__syncthreads();
		if (tx == 0) {
			for (int i = 0; i < 4; i++) {
				int row = 4 * ty + i;
				float current_score = best_score[row];
				int current_index = best_index[row];
				for (int col = 0; col < 32; col++) {
					float d = tile_score[row * 32 + col];
					if (d < current_score) {
						current_score = d;
						current_index = p2 + col;
					}
				}
				best_score[row] = current_score;
				best_index[row] = current_index;
			}
		}
		__syncthreads();
	}

	if (tx == 0) {
		for (int i = 0; i < 4; i++) {
			int row = 4 * ty + i;
			int p1 = p1_base + row;
			score[p1] = best_score[row];
			index[p1] = best_index[row];
		}
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
	for (const Descriptor& descriptor :  descriptors) {
		std::copy(descriptor.begin(), descriptor.end(), host_memory + offset);
		offset += descriptor.size();
	}
}

bool Match(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result) {

	size_t size = lhs.size();
	if (size == 0 || size != rhs.size() || size % 32 != 0) {
		fprintf(stderr, "ERROR Checked : invalid descriptor sizes\n");
		return false;
	}
	match_result.clear();

	float* lhs_descriptor_host = nullptr;
	float* rhs_descriptor_host = nullptr;
	float* lhs_descriptor_device = nullptr;
	float* rhs_descriptor_device = nullptr;
	float* device_score = nullptr;
	int* host_index = nullptr;
	int* device_index = nullptr;
	dim3 thread(32, 8);
	dim3 block(size / 32, 1);
	bool success = false;

	if (!CUDAErrorCheck(cudaMallocHost(&lhs_descriptor_host, sizeof(float) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMallocHost(&rhs_descriptor_host, sizeof(float) * size * 128))) goto cleanup;
	// copy the data
	CopyDescriptorToPinnedMemory(lhs, lhs_descriptor_host);
	CopyDescriptorToPinnedMemory(rhs, rhs_descriptor_host);

	if (!CUDAErrorCheck(cudaMalloc(&lhs_descriptor_device, sizeof(float) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&rhs_descriptor_device, sizeof(float) * size * 128))) goto cleanup;
	// copy from host to device
	if (!CUDAErrorCheck(cudaMemcpy(lhs_descriptor_device, lhs_descriptor_host, sizeof(float) * size * 128, cudaMemcpyHostToDevice))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(rhs_descriptor_device, rhs_descriptor_host, sizeof(float) * size * 128, cudaMemcpyHostToDevice))) goto cleanup;

	if (!CUDAErrorCheck(cudaMallocHost(&host_index, sizeof(int) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&device_score, sizeof(float) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&device_index, sizeof(int) * size))) goto cleanup;

	ComputeNearestNeighbor<<<block, thread>>>(lhs_descriptor_device, rhs_descriptor_device, device_score, device_index, size);
	if (!CUDAErrorCheck(cudaGetLastError())) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(host_index, device_index, sizeof(int) * size, cudaMemcpyDeviceToHost))) goto cleanup;

	for (int i = 0; i < size; i++) {
		std::pair<int, int> p(i, host_index[i]);
		match_result.push_back(p);
	}
	success = true;

cleanup:
	if (lhs_descriptor_host) cudaFreeHost(lhs_descriptor_host);
	if (rhs_descriptor_host) cudaFreeHost(rhs_descriptor_host);
	if (lhs_descriptor_device) cudaFree(lhs_descriptor_device);
	if (rhs_descriptor_device) cudaFree(rhs_descriptor_device);
	if (device_score) cudaFree(device_score);
	if (device_index) cudaFree(device_index);
	if (host_index) cudaFreeHost(host_index);
	if (!success) match_result.clear();
	return success;
}
