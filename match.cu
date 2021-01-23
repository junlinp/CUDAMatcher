#include "match.h"
#include "cuda_runtime.h"
#include "cuda_device_runtime_api.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "driver_types.h"
__global__ void PreliminaryComputeDistanceMatrix(float* pts1, float* pts2, float* distance_matrix, int WIDTH) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int p1_base = blockIdx.x * blockDim.x;
	int p1 = p1_base + tx;
	if (ty == 0) {
		
		for (int p2 = 0; p2 < WIDTH; p2++) {
			float* pts1_ptr = pts1 + p1 * 128;
			float* pts2_ptr = pts2 + p2 * 128;
			float d = 0.0;
			for (int k = 0; k < 128; k++) {
				float temp = pts1_ptr[k] - pts2_ptr[k];
				d += temp * temp;
			}
			distance_matrix[p1 * WIDTH + p2] = d;
		}
	}
}

__global__ void ComputeDistanceMatrix(float* pts1, float* pts2, float* distance_matrix, int WIDTH) {

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int p1_base = blockIdx.x * blockDim.x;
	int p1 = p1_base + tx;
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
					score[row + 32 * col] = ss[dx * 4 + dy];
				}
			}
		}
		__syncthreads();
		for (int i = 0; i < 4; i++) {
			rotation_score[(tx + ( 4 * ty + i)) % 32 + 32 * (4 * ty + i)] = score[tx + 32 * (4 * ty + i)];
		}
		__syncthreads();
		for (int i = 0; i < 4; i++) {
			distance_matrix[(p1_base + 4 * ty + i) * WIDTH + p2 + tx] = rotation_score[tx * 32 + (ty * 4 + i + tx) % 32];
		}
	}	
	
}

#define BATCH_NUM_ 256
__global__ void CloseElement(float* distance_matrix, int WIDTH, float* score, int* index) {
	int p1 = blockDim.x * blockIdx.x + threadIdx.x;
	int p1_base = blockDim.x * blockIdx.x;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int index_ = -1;
	float minimum_score = 1e10f;

	__shared__ float buffer_distance[32 * BATCH_NUM_];
	__shared__ int buffer_index[32 * 32];
	float d = 0.0;
	for (int i = 0; i < WIDTH; i += BATCH_NUM_) {
		for (int batch_index = 0; batch_index < BATCH_NUM_ / 32; batch_index++) {
			buffer_distance[ty * BATCH_NUM_ + (tx + ty) % 32 + batch_index * 32] = distance_matrix[(p1_base + ty) * WIDTH + i + tx + batch_index * 32];
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
				index_ =buffer_index[tx * 32 + (i + tx) % 32];

			}
		}
		score[p1] = minimum_score;
		index[p1] = index_;

		//printf("%d\n", p1);
	}
}

#define CUDAErrorCheck(x) ErrorCheckImp((x), __FILE__, __LINE__)

void ErrorCheckImp(cudaError_t code, const char* file_name, const int line) {
	if (code != cudaSuccess) {
		fprintf(stderr, "ERROR Checked : %s in %s %d\n", cudaGetErrorString(code), file_name, line);
		
	}
}
void CopyDescriptorToPinnedMemory(const std::vector<Descriptor>& descriptors, float* host_memory) {
	size_t offset = 0;
	for (const Descriptor& descriptor :  descriptors) {
		std::copy(descriptor.begin(), descriptor.end(), host_memory + offset);
		offset += descriptor.size();
	}
}

bool Match(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result) {

	// we assume that lhs and rhs are the same of size
	size_t size = lhs.size();

	float* lhs_descriptor_host = nullptr;
	float* rhs_descriptor_host = nullptr;
	float* lhs_descriptor_device = nullptr;
	float* rhs_descriptor_device = nullptr;

	CUDAErrorCheck(cudaMallocHost(&lhs_descriptor_host, sizeof(float) * size * 128));
	CUDAErrorCheck(cudaMallocHost(&rhs_descriptor_host, sizeof(float) * size * 128));
	// copy the data
	CopyDescriptorToPinnedMemory(lhs, lhs_descriptor_host);
	CopyDescriptorToPinnedMemory(rhs, rhs_descriptor_host);

	CUDAErrorCheck(cudaMalloc(&lhs_descriptor_device, sizeof(float) * size * 128));
	CUDAErrorCheck(cudaMalloc(&rhs_descriptor_device, sizeof(float) * size * 128));
	// copy from host to device
	CUDAErrorCheck(cudaMemcpy(lhs_descriptor_device, lhs_descriptor_host, sizeof(float) * size * 128, cudaMemcpyHostToDevice));
	CUDAErrorCheck(cudaMemcpy(rhs_descriptor_device, rhs_descriptor_host, sizeof(float) * size * 128, cudaMemcpyHostToDevice));



	float* distance_matrix = nullptr;
	CUDAErrorCheck(cudaMalloc(&distance_matrix, sizeof(float) * size * size));

	float* host_score = nullptr;
	float* device_score = nullptr;
	int* host_index = nullptr;
	int* device_index = nullptr;

	CUDAErrorCheck(cudaMallocHost(&host_score, sizeof(float) * size));
	CUDAErrorCheck(cudaMallocHost(&host_index, sizeof(int) * size));
	CUDAErrorCheck(cudaMalloc(&device_score, sizeof(float) * size));
	CUDAErrorCheck(cudaMalloc(&device_index, sizeof(int) * size));
	dim3 distance_thread(32, 8);
	dim3 distance_block(size / 32, 1);
	ComputeDistanceMatrix<<<distance_block, distance_thread>>>(lhs_descriptor_device, rhs_descriptor_device, distance_matrix, size);
	dim3 thread(32, 32);
	dim3 block(size / 32, 1);

	

	CloseElement<<<block, thread>>>(distance_matrix, size, device_score, device_index);

	CUDAErrorCheck(cudaMemcpy(host_index, device_index, sizeof(int) * size, cudaMemcpyDeviceToHost));
	CUDAErrorCheck(cudaMemcpy(host_score, device_score, sizeof(float) * size, cudaMemcpyDeviceToHost));

	for (int i = 0; i < size; i++) {
		std::pair<int, int> p(i, host_index[i]);
		match_result.push_back(p);
	}

	cudaFree(device_score);
	cudaFree(device_index);
	cudaFreeHost(host_score);
	cudaFreeHost(host_index);

	return true;
}
