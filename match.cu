#include "match.h"

#include <algorithm>
#include <cstdio>

#include "cublas_v2.h"
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

__device__ void WarpReduceMin(float& value, int& value_index) {
	for (int offset = 16; offset > 0; offset /= 2) {
		float other_value = __shfl_down_sync(0xffffffff, value, offset);
		int other_index = __shfl_down_sync(0xffffffff, value_index, offset);
		if (other_value < value || (other_value == value && other_index < value_index)) {
			value = other_value;
			value_index = other_index;
		}
	}
}

__global__ void ComputeNearestNeighborV2(float* pts1, float* pts2, float* score, int* index, int WIDTH) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int p1_base = blockIdx.x * blockDim.x;
	__shared__ float4 buffer_pts1[32 * 32];
	__shared__ float4 buffer_pts2[32 * 32];
	__shared__ float score_buffer[32 * 32];
	__shared__ float rotation_score[32 * 32];
	float best_score0 = 1e30f;
	float best_score1 = 1e30f;
	float best_score2 = 1e30f;
	float best_score3 = 1e30f;
	int best_index0 = -1;
	int best_index1 = -1;
	int best_index2 = -1;
	int best_index3 = -1;

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
					score_buffer[row + 32 * col] = ss[dx * 4 + dy];
				}
			}
		}
		__syncthreads();
		for (int i = 0; i < 4; i++) {
			rotation_score[(tx + (4 * ty + i)) % 32 + 32 * (4 * ty + i)] = score_buffer[tx + 32 * (4 * ty + i)];
		}
		__syncthreads();
		for (int i = 0; i < 4; i++) {
			int row = 4 * ty + i;
			float tile_best_score = rotation_score[tx * 32 + (row + tx) % 32];
			int tile_best_index = p2 + tx;
			WarpReduceMin(tile_best_score, tile_best_index);
			if (tx == 0) {
				if (i == 0 && tile_best_score < best_score0) {
					best_score0 = tile_best_score;
					best_index0 = tile_best_index;
				}
				if (i == 1 && tile_best_score < best_score1) {
					best_score1 = tile_best_score;
					best_index1 = tile_best_index;
				}
				if (i == 2 && tile_best_score < best_score2) {
					best_score2 = tile_best_score;
					best_index2 = tile_best_index;
				}
				if (i == 3 && tile_best_score < best_score3) {
					best_score3 = tile_best_score;
					best_index3 = tile_best_index;
				}
			}
		}
		__syncthreads();
	}

	if (tx == 0) {
		int p1 = p1_base + 4 * ty;
		score[p1 + 0] = best_score0;
		index[p1 + 0] = best_index0;
		score[p1 + 1] = best_score1;
		index[p1 + 1] = best_index1;
		score[p1 + 2] = best_score2;
		index[p1 + 2] = best_index2;
		score[p1 + 3] = best_score3;
		index[p1 + 3] = best_index3;
	}
}

#define CUDAErrorCheck(x) ErrorCheckImp((x), __FILE__, __LINE__)
#define CUBLASErrorCheck(x) CublasErrorCheckImp((x), __FILE__, __LINE__)

bool ErrorCheckImp(cudaError_t code, const char* file_name, const int line) {
	if (code != cudaSuccess) {
		fprintf(stderr, "ERROR Checked : %s in %s %d\n", cudaGetErrorString(code), file_name, line);
		return false;
	}
	return true;
}

bool CublasErrorCheckImp(cublasStatus_t code, const char* file_name, const int line) {
	if (code != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "ERROR Checked : cuBLAS status %d in %s %d\n", static_cast<int>(code), file_name, line);
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

void CopyDescriptorToPinnedMemoryAndNorm(const std::vector<Descriptor>& descriptors, float* host_memory, float* norms) {
	size_t offset = 0;
	size_t norm_index = 0;
	for (const Descriptor& descriptor : descriptors) {
		float norm = 0.0f;
		for (float value : descriptor) {
			host_memory[offset++] = value;
			norm += value * value;
		}
		norms[norm_index++] = norm;
	}
}

__global__ void InitializeBestKernel(float* best_score, int* best_index, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		best_score[idx] = 1e30f;
		best_index[idx] = -1;
	}
}

__global__ void ReduceDotTileKernel(const float* dot_tile,
                                    const float* lhs_norm,
                                    const float* rhs_norm,
                                    float* best_score,
                                    int* best_index,
                                    int width,
                                    int tile_start,
                                    int tile_size) {
	int lhs_index = blockIdx.x;
	int lane = threadIdx.x;
	float local_best = 1e30f;
	int local_index = -1;

	for (int rhs_offset = lane; rhs_offset < tile_size; rhs_offset += 32) {
		float dot = dot_tile[rhs_offset + lhs_index * tile_size];
		float distance = lhs_norm[lhs_index] + rhs_norm[tile_start + rhs_offset] - 2.0f * dot;
		if (distance < local_best || (distance == local_best && tile_start + rhs_offset < local_index)) {
			local_best = distance;
			local_index = tile_start + rhs_offset;
		}
	}

	WarpReduceMin(local_best, local_index);
	if (lane == 0 && local_best < best_score[lhs_index]) {
		best_score[lhs_index] = local_best;
		best_index[lhs_index] = local_index;
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

bool MatchV2(const std::vector<Descriptor>& lhs,
             const std::vector<Descriptor>& rhs,
             std::vector<std::pair<int, int>>& match_result) {
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
	CopyDescriptorToPinnedMemory(lhs, lhs_descriptor_host);
	CopyDescriptorToPinnedMemory(rhs, rhs_descriptor_host);

	if (!CUDAErrorCheck(cudaMalloc(&lhs_descriptor_device, sizeof(float) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&rhs_descriptor_device, sizeof(float) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(lhs_descriptor_device, lhs_descriptor_host, sizeof(float) * size * 128, cudaMemcpyHostToDevice))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(rhs_descriptor_device, rhs_descriptor_host, sizeof(float) * size * 128, cudaMemcpyHostToDevice))) goto cleanup;

	if (!CUDAErrorCheck(cudaMallocHost(&host_index, sizeof(int) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&device_score, sizeof(float) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&device_index, sizeof(int) * size))) goto cleanup;

	ComputeNearestNeighborV2<<<block, thread>>>(lhs_descriptor_device, rhs_descriptor_device, device_score, device_index, size);
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
	if (device_score) cudaFree(device_score);
	if (device_index) cudaFree(device_index);
	if (host_index) cudaFreeHost(host_index);
	if (!success) match_result.clear();
	return success;
}

bool MatchV3(const std::vector<Descriptor>& lhs,
             const std::vector<Descriptor>& rhs,
             std::vector<std::pair<int, int>>& match_result) {
	size_t size = lhs.size();
	const int kDescriptorLength = 128;
	const int kTileSize = 1024;
	if (size == 0 || size != rhs.size() || size % kTileSize != 0) {
		fprintf(stderr, "ERROR Checked : invalid descriptor sizes\n");
		return false;
	}
	match_result.clear();

	float* lhs_descriptor_host = nullptr;
	float* rhs_descriptor_host = nullptr;
	float* lhs_norm_host = nullptr;
	float* rhs_norm_host = nullptr;
	float* lhs_descriptor_device = nullptr;
	float* rhs_descriptor_device = nullptr;
	float* lhs_norm_device = nullptr;
	float* rhs_norm_device = nullptr;
	float* dot_tile_device = nullptr;
	float* device_score = nullptr;
	int* host_index = nullptr;
	int* device_index = nullptr;
	cublasHandle_t handle = nullptr;
	bool success = false;
	const float alpha = 1.0f;
	const float beta = 0.0f;

	if (!CUDAErrorCheck(cudaMallocHost(&lhs_descriptor_host, sizeof(float) * size * kDescriptorLength))) goto cleanup;
	if (!CUDAErrorCheck(cudaMallocHost(&rhs_descriptor_host, sizeof(float) * size * kDescriptorLength))) goto cleanup;
	if (!CUDAErrorCheck(cudaMallocHost(&lhs_norm_host, sizeof(float) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMallocHost(&rhs_norm_host, sizeof(float) * size))) goto cleanup;
	CopyDescriptorToPinnedMemoryAndNorm(lhs, lhs_descriptor_host, lhs_norm_host);
	CopyDescriptorToPinnedMemoryAndNorm(rhs, rhs_descriptor_host, rhs_norm_host);

	if (!CUDAErrorCheck(cudaMalloc(&lhs_descriptor_device, sizeof(float) * size * kDescriptorLength))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&rhs_descriptor_device, sizeof(float) * size * kDescriptorLength))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&lhs_norm_device, sizeof(float) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&rhs_norm_device, sizeof(float) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&dot_tile_device, sizeof(float) * size * kTileSize))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&device_score, sizeof(float) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&device_index, sizeof(int) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMallocHost(&host_index, sizeof(int) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(lhs_descriptor_device, lhs_descriptor_host, sizeof(float) * size * kDescriptorLength, cudaMemcpyHostToDevice))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(rhs_descriptor_device, rhs_descriptor_host, sizeof(float) * size * kDescriptorLength, cudaMemcpyHostToDevice))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(lhs_norm_device, lhs_norm_host, sizeof(float) * size, cudaMemcpyHostToDevice))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(rhs_norm_device, rhs_norm_host, sizeof(float) * size, cudaMemcpyHostToDevice))) goto cleanup;
	if (!CUBLASErrorCheck(cublasCreate(&handle))) goto cleanup;

	InitializeBestKernel<<<static_cast<unsigned int>((size + 255) / 256), 256>>>(device_score, device_index, size);
	if (!CUDAErrorCheck(cudaGetLastError())) goto cleanup;

	for (int tile_start = 0; tile_start < static_cast<int>(size); tile_start += kTileSize) {
		if (!CUBLASErrorCheck(cublasSgemm(handle,
		                                  CUBLAS_OP_T,
		                                  CUBLAS_OP_N,
		                                  kTileSize,
		                                  static_cast<int>(size),
		                                  kDescriptorLength,
		                                  &alpha,
		                                  rhs_descriptor_device + tile_start * kDescriptorLength,
		                                  kDescriptorLength,
		                                  lhs_descriptor_device,
		                                  kDescriptorLength,
		                                  &beta,
		                                  dot_tile_device,
		                                  kTileSize))) goto cleanup;
		ReduceDotTileKernel<<<static_cast<unsigned int>(size), 32>>>(dot_tile_device,
		                                                             lhs_norm_device,
		                                                             rhs_norm_device,
		                                                             device_score,
		                                                             device_index,
		                                                             static_cast<int>(size),
		                                                             tile_start,
		                                                             kTileSize);
		if (!CUDAErrorCheck(cudaGetLastError())) goto cleanup;
	}

	if (!CUDAErrorCheck(cudaMemcpy(host_index, device_index, sizeof(int) * size, cudaMemcpyDeviceToHost))) goto cleanup;
	for (size_t i = 0; i < size; i++) {
		match_result.push_back(std::pair<int, int>(static_cast<int>(i), host_index[i]));
	}
	success = true;

cleanup:
	if (handle) cublasDestroy(handle);
	if (lhs_descriptor_host) cudaFreeHost(lhs_descriptor_host);
	if (rhs_descriptor_host) cudaFreeHost(rhs_descriptor_host);
	if (lhs_norm_host) cudaFreeHost(lhs_norm_host);
	if (rhs_norm_host) cudaFreeHost(rhs_norm_host);
	if (lhs_descriptor_device) cudaFree(lhs_descriptor_device);
	if (rhs_descriptor_device) cudaFree(rhs_descriptor_device);
	if (lhs_norm_device) cudaFree(lhs_norm_device);
	if (rhs_norm_device) cudaFree(rhs_norm_device);
	if (dot_tile_device) cudaFree(dot_tile_device);
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

bool BenchmarkKernelV2(const std::vector<Descriptor>& lhs,
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
	if (size == 0 || size != rhs.size() || size != expected_match.size() || size % 32 != 0 || warmup_runs < 0 || measured_runs <= 0) {
		fprintf(stderr, "ERROR Checked : invalid kernel benchmark input\n");
		return false;
	}

	float* lhs_descriptor_host = nullptr;
	float* rhs_descriptor_host = nullptr;
	float* lhs_descriptor_device = nullptr;
	float* rhs_descriptor_device = nullptr;
	float* device_score = nullptr;
	int* host_index = nullptr;
	int* device_index = nullptr;
	cudaEvent_t start = nullptr;
	cudaEvent_t stop = nullptr;
	dim3 thread(32, 8);
	dim3 block(size / 32, 1);
	bool success = false;
	float best_ms = 1e30f;
	float sum_ms = 0.0f;

	if (!CUDAErrorCheck(cudaMallocHost(&lhs_descriptor_host, sizeof(float) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMallocHost(&rhs_descriptor_host, sizeof(float) * size * 128))) goto cleanup;
	CopyDescriptorToPinnedMemory(lhs, lhs_descriptor_host);
	CopyDescriptorToPinnedMemory(rhs, rhs_descriptor_host);
	if (!CUDAErrorCheck(cudaMalloc(&lhs_descriptor_device, sizeof(float) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&rhs_descriptor_device, sizeof(float) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&device_score, sizeof(float) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&device_index, sizeof(int) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMallocHost(&host_index, sizeof(int) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(lhs_descriptor_device, lhs_descriptor_host, sizeof(float) * size * 128, cudaMemcpyHostToDevice))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(rhs_descriptor_device, rhs_descriptor_host, sizeof(float) * size * 128, cudaMemcpyHostToDevice))) goto cleanup;

	for (int i = 0; i < warmup_runs; i++) {
		ComputeNearestNeighborV2<<<block, thread>>>(lhs_descriptor_device, rhs_descriptor_device, device_score, device_index, size);
	}
	if (!CUDAErrorCheck(cudaGetLastError())) goto cleanup;
	if (!CUDAErrorCheck(cudaDeviceSynchronize())) goto cleanup;

	if (!CUDAErrorCheck(cudaEventCreate(&start))) goto cleanup;
	if (!CUDAErrorCheck(cudaEventCreate(&stop))) goto cleanup;
	for (int i = 0; i < measured_runs; i++) {
		float elapsed_ms = 0.0f;
		if (!CUDAErrorCheck(cudaEventRecord(start, 0))) goto cleanup;
		ComputeNearestNeighborV2<<<block, thread>>>(lhs_descriptor_device, rhs_descriptor_device, device_score, device_index, size);
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
	if (device_score) cudaFree(device_score);
	if (device_index) cudaFree(device_index);
	if (host_index) cudaFreeHost(host_index);
	return success;
}

bool BenchmarkKernelV3(const std::vector<Descriptor>& lhs,
                       const std::vector<Descriptor>& rhs,
                       const std::vector<int>& expected_match,
                       int warmup_runs,
                       int measured_runs,
                       KernelBenchmarkResult& result) {
	size_t size = lhs.size();
	const int kDescriptorLength = 128;
	const int kTileSize = 1024;
	result.success = false;
	result.best_ms = 0.0f;
	result.avg_ms = 0.0f;
	result.mismatch_count = static_cast<int>(expected_match.size());
	if (size == 0 || size != rhs.size() || size != expected_match.size() || size % kTileSize != 0 || warmup_runs < 0 || measured_runs <= 0) {
		fprintf(stderr, "ERROR Checked : invalid kernel benchmark input\n");
		return false;
	}

	float* lhs_descriptor_host = nullptr;
	float* rhs_descriptor_host = nullptr;
	float* lhs_norm_host = nullptr;
	float* rhs_norm_host = nullptr;
	float* lhs_descriptor_device = nullptr;
	float* rhs_descriptor_device = nullptr;
	float* lhs_norm_device = nullptr;
	float* rhs_norm_device = nullptr;
	float* dot_tile_device = nullptr;
	float* device_score = nullptr;
	int* host_index = nullptr;
	int* device_index = nullptr;
	cublasHandle_t handle = nullptr;
	cudaEvent_t start = nullptr;
	cudaEvent_t stop = nullptr;
	bool success = false;
	float best_ms = 1e30f;
	float sum_ms = 0.0f;
	const float alpha = 1.0f;
	const float beta = 0.0f;

	if (!CUDAErrorCheck(cudaMallocHost(&lhs_descriptor_host, sizeof(float) * size * kDescriptorLength))) goto cleanup;
	if (!CUDAErrorCheck(cudaMallocHost(&rhs_descriptor_host, sizeof(float) * size * kDescriptorLength))) goto cleanup;
	if (!CUDAErrorCheck(cudaMallocHost(&lhs_norm_host, sizeof(float) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMallocHost(&rhs_norm_host, sizeof(float) * size))) goto cleanup;
	CopyDescriptorToPinnedMemoryAndNorm(lhs, lhs_descriptor_host, lhs_norm_host);
	CopyDescriptorToPinnedMemoryAndNorm(rhs, rhs_descriptor_host, rhs_norm_host);
	if (!CUDAErrorCheck(cudaMalloc(&lhs_descriptor_device, sizeof(float) * size * kDescriptorLength))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&rhs_descriptor_device, sizeof(float) * size * kDescriptorLength))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&lhs_norm_device, sizeof(float) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&rhs_norm_device, sizeof(float) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&dot_tile_device, sizeof(float) * size * kTileSize))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&device_score, sizeof(float) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&device_index, sizeof(int) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMallocHost(&host_index, sizeof(int) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(lhs_descriptor_device, lhs_descriptor_host, sizeof(float) * size * kDescriptorLength, cudaMemcpyHostToDevice))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(rhs_descriptor_device, rhs_descriptor_host, sizeof(float) * size * kDescriptorLength, cudaMemcpyHostToDevice))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(lhs_norm_device, lhs_norm_host, sizeof(float) * size, cudaMemcpyHostToDevice))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(rhs_norm_device, rhs_norm_host, sizeof(float) * size, cudaMemcpyHostToDevice))) goto cleanup;
	if (!CUBLASErrorCheck(cublasCreate(&handle))) goto cleanup;

	for (int run = 0; run < warmup_runs; run++) {
		InitializeBestKernel<<<static_cast<unsigned int>((size + 255) / 256), 256>>>(device_score, device_index, size);
		for (int tile_start = 0; tile_start < static_cast<int>(size); tile_start += kTileSize) {
			if (!CUBLASErrorCheck(cublasSgemm(handle,
			                                  CUBLAS_OP_T,
			                                  CUBLAS_OP_N,
			                                  kTileSize,
			                                  static_cast<int>(size),
			                                  kDescriptorLength,
			                                  &alpha,
			                                  rhs_descriptor_device + tile_start * kDescriptorLength,
			                                  kDescriptorLength,
			                                  lhs_descriptor_device,
			                                  kDescriptorLength,
			                                  &beta,
			                                  dot_tile_device,
			                                  kTileSize))) goto cleanup;
			ReduceDotTileKernel<<<static_cast<unsigned int>(size), 32>>>(dot_tile_device,
			                                                             lhs_norm_device,
			                                                             rhs_norm_device,
			                                                             device_score,
			                                                             device_index,
			                                                             static_cast<int>(size),
			                                                             tile_start,
			                                                             kTileSize);
		}
	}
	if (!CUDAErrorCheck(cudaGetLastError())) goto cleanup;
	if (!CUDAErrorCheck(cudaDeviceSynchronize())) goto cleanup;

	if (!CUDAErrorCheck(cudaEventCreate(&start))) goto cleanup;
	if (!CUDAErrorCheck(cudaEventCreate(&stop))) goto cleanup;
	for (int run = 0; run < measured_runs; run++) {
		float elapsed_ms = 0.0f;
		if (!CUDAErrorCheck(cudaEventRecord(start, 0))) goto cleanup;
		InitializeBestKernel<<<static_cast<unsigned int>((size + 255) / 256), 256>>>(device_score, device_index, size);
		for (int tile_start = 0; tile_start < static_cast<int>(size); tile_start += kTileSize) {
			if (!CUBLASErrorCheck(cublasSgemm(handle,
			                                  CUBLAS_OP_T,
			                                  CUBLAS_OP_N,
			                                  kTileSize,
			                                  static_cast<int>(size),
			                                  kDescriptorLength,
			                                  &alpha,
			                                  rhs_descriptor_device + tile_start * kDescriptorLength,
			                                  kDescriptorLength,
			                                  lhs_descriptor_device,
			                                  kDescriptorLength,
			                                  &beta,
			                                  dot_tile_device,
			                                  kTileSize))) goto cleanup;
			ReduceDotTileKernel<<<static_cast<unsigned int>(size), 32>>>(dot_tile_device,
			                                                             lhs_norm_device,
			                                                             rhs_norm_device,
			                                                             device_score,
			                                                             device_index,
			                                                             static_cast<int>(size),
			                                                             tile_start,
			                                                             kTileSize);
		}
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
	if (handle) cublasDestroy(handle);
	if (lhs_descriptor_host) cudaFreeHost(lhs_descriptor_host);
	if (rhs_descriptor_host) cudaFreeHost(rhs_descriptor_host);
	if (lhs_norm_host) cudaFreeHost(lhs_norm_host);
	if (rhs_norm_host) cudaFreeHost(rhs_norm_host);
	if (lhs_descriptor_device) cudaFree(lhs_descriptor_device);
	if (rhs_descriptor_device) cudaFree(rhs_descriptor_device);
	if (lhs_norm_device) cudaFree(lhs_norm_device);
	if (rhs_norm_device) cudaFree(rhs_norm_device);
	if (dot_tile_device) cudaFree(dot_tile_device);
	if (device_score) cudaFree(device_score);
	if (device_index) cudaFree(device_index);
	if (host_index) cudaFreeHost(host_index);
	return success;
}

bool Match(const std::vector<Descriptor>& lhs,
           const std::vector<Descriptor>& rhs,
           std::vector<std::pair<int, int>>& match_result) {
	return MatchV3(lhs, rhs, match_result);
}
