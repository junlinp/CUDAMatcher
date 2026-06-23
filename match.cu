#include "match.h"
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <vector>
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include <mma.h>

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
__global__ void CloseElementV1(float* distance_matrix, int WIDTH, float* score, int* index) {
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
	}
}

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

__global__ void ComputeNearestNeighborV3(float* pts1, float* pts2, float* score, int* index, int WIDTH) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int p1_base = blockIdx.x * blockDim.x;
	__shared__ float4 buffer_pts1[32 * 32];
	__shared__ float4 buffer_pts2[32 * 32];
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
			float tile_best_score = rotation_score[tx * 32 + (row + tx) % 32];
			int tile_best_index = p2 + tx;
			WarpReduceMin(tile_best_score, tile_best_index);
			if (tx == 0 && tile_best_score < best_score[row]) {
				best_score[row] = tile_best_score;
				best_index[row] = tile_best_index;
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

__global__ void ComputeNearestNeighborV5a(float* pts1, float* pts2, float* score, int* index, int WIDTH) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int p1_base = blockIdx.x * blockDim.x;
	__shared__ float4 buffer_pts1[32 * 32];
	__shared__ float4 buffer_pts2[32 * 32];
	__shared__ float score_buffer[32 * 32];
	__shared__ float rotation_score[32 * 32];
	__shared__ float best_score[32];
	__shared__ int best_index[32];

	const float4* pts1_vec = (const float4*)pts1;
	const float4* pts2_vec = (const float4*)pts2;

	for (int i = 0; i < 4; i++) {
		buffer_pts1[(ty * 4 + i) * 32 + tx] = pts1_vec[(p1_base + ty * 4 + i) * 32 + tx];
		if (tx == 0) {
			int row = 4 * ty + i;
			best_score[row] = 1e30f;
			best_index[row] = -1;
		}
	}
	__syncthreads();

	for (int p2 = 0; p2 < WIDTH; p2 += 32) {
		for (int i = 0; i < 4; i++) {
			buffer_pts2[(ty * 4 + i) * 32 + tx] = pts2_vec[(p2 + ty * 4 + i) * 32 + tx];
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
			float tile_best_score = rotation_score[tx * 32 + (row + tx) % 32];
			int tile_best_index = p2 + tx;
			WarpReduceMin(tile_best_score, tile_best_index);
			if (tx == 0 && tile_best_score < best_score[row]) {
				best_score[row] = tile_best_score;
				best_index[row] = tile_best_index;
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

__global__ void ComputeNearestNeighborV5c(float* pts1, float* pts2, float* score, int* index, int WIDTH) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int p1_base = blockIdx.x * blockDim.x;
	const int p1_start = p1_base + ty * 4;

	__shared__ float4 buffer_pts1[32 * 32];
	__shared__ float4 buffer_pts2[32 * 32];

	const float4* pts1_vec = (const float4*)pts1;
	const float4* pts2_vec = (const float4*)pts2;

	#pragma unroll
	for (int i = 0; i < 4; i++) {
		buffer_pts1[(ty * 4 + i) * 32 + tx] = pts1_vec[(p1_start + i) * 32 + tx];
	}
	__syncthreads();

	float best0 = 1e30f;
	float best1 = 1e30f;
	float best2 = 1e30f;
	float best3 = 1e30f;
	int best_index0 = -1;
	int best_index1 = -1;
	int best_index2 = -1;
	int best_index3 = -1;

	for (int p2_base = 0; p2_base < WIDTH; p2_base += 32) {
		int linear = threadIdx.x + threadIdx.y * blockDim.x;
		for (int item = linear; item < 32 * 32; item += blockDim.x * blockDim.y) {
			int p2_offset = item / 32;
			int k = item % 32;
			buffer_pts2[item] = pts2_vec[(p2_base + p2_offset) * 32 + k];
		}
		__syncthreads();

		float d0 = 0.0f;
		float d1 = 0.0f;
		float d2 = 0.0f;
		float d3 = 0.0f;
		if (p2_base + tx < WIDTH) {
			for (int k = 0; k < 32; k++) {
				float4 lhs0 = buffer_pts1[(ty * 4 + 0) * 32 + k];
				float4 lhs1 = buffer_pts1[(ty * 4 + 1) * 32 + k];
				float4 lhs2 = buffer_pts1[(ty * 4 + 2) * 32 + k];
				float4 lhs3 = buffer_pts1[(ty * 4 + 3) * 32 + k];
				float4 rhs0 = buffer_pts2[tx * 32 + k];

				float a = lhs0.x - rhs0.x;
				d0 += a * a;
				a = lhs0.y - rhs0.y;
				d0 += a * a;
				a = lhs0.z - rhs0.z;
				d0 += a * a;
				a = lhs0.w - rhs0.w;
				d0 += a * a;

				a = lhs1.x - rhs0.x;
				d1 += a * a;
				a = lhs1.y - rhs0.y;
				d1 += a * a;
				a = lhs1.z - rhs0.z;
				d1 += a * a;
				a = lhs1.w - rhs0.w;
				d1 += a * a;

				a = lhs2.x - rhs0.x;
				d2 += a * a;
				a = lhs2.y - rhs0.y;
				d2 += a * a;
				a = lhs2.z - rhs0.z;
				d2 += a * a;
				a = lhs2.w - rhs0.w;
				d2 += a * a;

				a = lhs3.x - rhs0.x;
				d3 += a * a;
				a = lhs3.y - rhs0.y;
				d3 += a * a;
				a = lhs3.z - rhs0.z;
				d3 += a * a;
				a = lhs3.w - rhs0.w;
				d3 += a * a;
			}
		}

		int tile_index = p2_base + tx;
		float tile_best0 = d0;
		float tile_best1 = d1;
		float tile_best2 = d2;
		float tile_best3 = d3;
		int tile_index0 = tile_index;
		int tile_index1 = tile_index;
		int tile_index2 = tile_index;
		int tile_index3 = tile_index;
		if (p2_base + tx >= WIDTH) {
			tile_best0 = 1e30f;
			tile_best1 = 1e30f;
			tile_best2 = 1e30f;
			tile_best3 = 1e30f;
			tile_index0 = -1;
			tile_index1 = -1;
			tile_index2 = -1;
			tile_index3 = -1;
		}

		WarpReduceMin(tile_best0, tile_index0);
		WarpReduceMin(tile_best1, tile_index1);
		WarpReduceMin(tile_best2, tile_index2);
		WarpReduceMin(tile_best3, tile_index3);
		if (tx == 0) {
			if (tile_best0 < best0) {
				best0 = tile_best0;
				best_index0 = tile_index0;
			}
			if (tile_best1 < best1) {
				best1 = tile_best1;
				best_index1 = tile_index1;
			}
			if (tile_best2 < best2) {
				best2 = tile_best2;
				best_index2 = tile_index2;
			}
			if (tile_best3 < best3) {
				best3 = tile_best3;
				best_index3 = tile_index3;
			}
		}
		__syncthreads();
	}

	if (tx == 0) {
		if (p1_start + 0 < WIDTH) {
			score[p1_start + 0] = best0;
			index[p1_start + 0] = best_index0;
		}
		if (p1_start + 1 < WIDTH) {
			score[p1_start + 1] = best1;
			index[p1_start + 1] = best_index1;
		}
		if (p1_start + 2 < WIDTH) {
			score[p1_start + 2] = best2;
			index[p1_start + 2] = best_index2;
		}
		if (p1_start + 3 < WIDTH) {
			score[p1_start + 3] = best3;
			index[p1_start + 3] = best_index3;
		}
	}
}

__global__ void ComputeNearestNeighborV5b(float* pts1, float* pts2, float* score, int* index, int WIDTH) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int p1_base = blockIdx.x * blockDim.x;
	const float4* pts1_vec = (const float4*)pts1;
	const float4* pts2_vec = (const float4*)pts2;

	__shared__ float4 buffer_pts1[32 * 32];
	__shared__ float4 buffer_pts2[32 * 32];
	__shared__ float score_buffer[32 * 32];
	__shared__ float rotation_score[32 * 32];
	__shared__ float best_score[32];
	__shared__ int best_index[32];

	for (int i = 0; i < 4; i++) {
		buffer_pts1[(ty * 4 + i) * 32 + tx] = pts1_vec[(p1_base + ty * 4 + i) * 32 + tx];
		if (tx == 0) {
			int row = 4 * ty + i;
			best_score[row] = 1e30f;
			best_index[row] = -1;
		}
	}
	__syncthreads();

	for (int p2 = 0; p2 < WIDTH; p2 += 32) {
		for (int i = 0; i < 4; i++) {
			buffer_pts2[(ty * 4 + i) * 32 + tx] = pts2_vec[(p2 + ty * 4 + i) * 32 + tx];
		}
		__syncthreads();
		if (ty < 4) {
			float ss[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
			for (int i = 0; i < 32; i++) {
				float4 v1[2];
				#pragma unroll
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
			if (tx == 0 && tile_best_score < best_score[row]) {
				best_score[row] = tile_best_score;
				best_index[row] = tile_best_index;
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

__global__ void ComputeNearestNeighborV4(float* pts1, float* pts2, float* score, int* index, int WIDTH) {
	int lane = threadIdx.x & 31;
	int warp = threadIdx.x >> 5;
	int p1 = blockIdx.x * 8 + warp;
	if (p1 >= WIDTH) {
		return;
	}

	__shared__ float4 buffer_pts1[8 * 32];
	__shared__ float4 buffer_pts2[32 * 32];
	float best_score = 1e30f;
	int best_index = -1;
	float4* pts1_vec = (float4*)pts1;
	float4* pts2_vec = (float4*)pts2;

	for (int k = lane; k < 32; k += 32) {
		buffer_pts1[warp * 32 + k] = pts1_vec[p1 * 32 + k];
	}
	__syncthreads();

	for (int p2_base = 0; p2_base < WIDTH; p2_base += 32) {
		for (int item = threadIdx.x; item < 32 * 32; item += blockDim.x) {
			int p2_offset = item / 32;
			int k = item % 32;
			buffer_pts2[p2_offset * 32 + k] = pts2_vec[(p2_base + p2_offset) * 32 + k];
		}
		__syncthreads();

		float d = 0.0f;
		for (int k = 0; k < 32; k++) {
			float4 v1 = buffer_pts1[warp * 32 + k];
			float4 v2 = buffer_pts2[lane * 32 + k];
			float a = v1.x - v2.x;
			d += a * a;
			a = v1.y - v2.y;
			d += a * a;
			a = v1.z - v2.z;
			d += a * a;
			a = v1.w - v2.w;
			d += a * a;
		}
		if (d < best_score) {
			best_score = d;
			best_index = p2_base + lane;
		}
		__syncthreads();
	}

	WarpReduceMin(best_score, best_index);
	if (lane == 0) {
		score[p1] = best_score;
		index[p1] = best_index;
	}
}

__global__ void ComputeNearestNeighborV6(const float* pts1, const float* pts2, float* score, int* index, int WIDTH) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int p1_base = blockIdx.x * blockDim.x;
	const float4* pts1_vec = (const float4*)pts1;
	const float4* pts2_vec = (const float4*)pts2;

	__shared__ float4 buffer_pts1[32 * 32];
	__shared__ float4 buffer_pts2[32 * 32];
	__shared__ float score_buffer[32 * 32];
	__shared__ float rotation_score[32 * 32];
	__shared__ float best_score[32];
	__shared__ int best_index[32];

	for (int i = 0; i < 4; i++) {
		buffer_pts1[(ty * 4 + i) * 32 + tx] = pts1_vec[(p1_base + ty * 4 + i) * 32 + tx];
		if (tx == 0) {
			int row = 4 * ty + i;
			best_score[row] = 1e30f;
			best_index[row] = -1;
		}
	}
	__syncthreads();

	for (int p2 = 0; p2 < WIDTH; p2 += 32) {
		for (int i = 0; i < 4; i++) {
			buffer_pts2[(ty * 4 + i) * 32 + tx] = pts2_vec[(p2 + ty * 4 + i) * 32 + tx];
		}
		__syncthreads();
		if (ty < 4) {
			float ss[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
			for (int i = 0; i < 32; i++) {
				float4 v1[2];
				#pragma unroll
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
			if (tx == 0 && tile_best_score < best_score[row]) {
				best_score[row] = tile_best_score;
				best_index[row] = tile_best_index;
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

__global__ void ComputeNearestNeighborV7(const float* pts1, const float* pts2, float* score, int* index, int WIDTH) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int p1_base = blockIdx.x * blockDim.x;
	const float4* pts1_vec = (const float4*)pts1;
	const float4* pts2_vec = (const float4*)pts2;

	__shared__ float4 buffer_pts1[32 * 32];
	__shared__ float4 buffer_pts2[32 * 32];
	__shared__ float score_buffer[32 * 32];
	__shared__ float rotation_score[32 * 32];

	float best0 = 1e30f;
	float best1 = 1e30f;
	float best2 = 1e30f;
	float best3 = 1e30f;
	int best_index0 = -1;
	int best_index1 = -1;
	int best_index2 = -1;
	int best_index3 = -1;

	for (int i = 0; i < 4; i++) {
		buffer_pts1[(ty * 4 + i) * 32 + tx] = pts1_vec[(p1_base + ty * 4 + i) * 32 + tx];
	}
	__syncthreads();

	for (int p2 = 0; p2 < WIDTH; p2 += 32) {
		for (int i = 0; i < 4; i++) {
			buffer_pts2[(ty * 4 + i) * 32 + tx] = pts2_vec[(p2 + ty * 4 + i) * 32 + tx];
		}
		__syncthreads();
		if (ty < 4) {
			float ss[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
			for (int i = 0; i < 32; i++) {
				float4 v1[2];
				#pragma unroll
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
					int col = 4 * (2 * ty + tx / 16) + dy;
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
				if (i == 0 && tile_best_score < best0) {
					best0 = tile_best_score;
					best_index0 = tile_best_index;
				} else if (i == 1 && tile_best_score < best1) {
					best1 = tile_best_score;
					best_index1 = tile_best_index;
				} else if (i == 2 && tile_best_score < best2) {
					best2 = tile_best_score;
					best_index2 = tile_best_index;
				} else if (i == 3 && tile_best_score < best3) {
					best3 = tile_best_score;
					best_index3 = tile_best_index;
				}
			}
		}
		__syncthreads();
	}

	if (tx == 0) {
		int p1_start = p1_base + ty * 4;
		score[p1_start + 0] = best0;
		index[p1_start + 0] = best_index0;
		score[p1_start + 1] = best1;
		index[p1_start + 1] = best_index1;
		score[p1_start + 2] = best2;
		index[p1_start + 2] = best_index2;
		score[p1_start + 3] = best3;
		index[p1_start + 3] = best_index3;
	}
}

__global__ void ComputeNearestNeighborV8(const __half* pts1, const __half* pts2, float* score, int* index, int WIDTH) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int p1_base = blockIdx.x * blockDim.x;
	const __half2* pts1_vec = (const __half2*)pts1;
	const __half2* pts2_vec = (const __half2*)pts2;

	__shared__ __half2 buffer_pts1[32 * 64];
	__shared__ __half2 buffer_pts2[32 * 64];
	__shared__ float score_buffer[32 * 32];
	__shared__ float rotation_score[32 * 32];

	float best0 = 1e30f;
	float best1 = 1e30f;
	float best2 = 1e30f;
	float best3 = 1e30f;
	int best_index0 = -1;
	int best_index1 = -1;
	int best_index2 = -1;
	int best_index3 = -1;

	for (int i = 0; i < 4; i++) {
		int row = ty * 4 + i;
		buffer_pts1[row * 64 + tx] = pts1_vec[(p1_base + row) * 64 + tx];
		buffer_pts1[row * 64 + tx + 32] = pts1_vec[(p1_base + row) * 64 + tx + 32];
	}
	__syncthreads();

	for (int p2 = 0; p2 < WIDTH; p2 += 32) {
		for (int i = 0; i < 4; i++) {
			int row = ty * 4 + i;
			buffer_pts2[row * 64 + tx] = pts2_vec[(p2 + row) * 64 + tx];
			buffer_pts2[row * 64 + tx + 32] = pts2_vec[(p2 + row) * 64 + tx + 32];
		}
		__syncthreads();
		if (ty < 4) {
			float ss[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
			for (int i = 0; i < 64; i++) {
				__half2 v1[2];
				#pragma unroll
				for (int dx = 0; dx < 2; dx++) {
					v1[dx] = buffer_pts1[((16 * dx + tx) % 32) * 64 + ((i + tx) % 64)];
				}

				for (int dy = 0; dy < 4; dy++) {
					__half2 pt2 = buffer_pts2[(4 * (2 * ty + tx / 16) + dy) * 64 + ((i + tx) % 64)];
					float2 rhs2 = __half22float2(pt2);
					float2 lhs0 = __half22float2(v1[0]);
					float2 lhs1 = __half22float2(v1[1]);

					float a = rhs2.x - lhs0.x;
					ss[dy] += a * a;
					a = rhs2.y - lhs0.y;
					ss[dy] += a * a;

					a = rhs2.x - lhs1.x;
					ss[4 + dy] += a * a;
					a = rhs2.y - lhs1.y;
					ss[4 + dy] += a * a;
				}
			}
			for (int dy = 0; dy < 4; dy++) {
				for (int dx = 0; dx < 2; dx++) {
					int row = (tx + 16 * dx) % 32;
					int col = 4 * (2 * ty + tx / 16) + dy;
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
				if (i == 0 && tile_best_score < best0) {
					best0 = tile_best_score;
					best_index0 = tile_best_index;
				} else if (i == 1 && tile_best_score < best1) {
					best1 = tile_best_score;
					best_index1 = tile_best_index;
				} else if (i == 2 && tile_best_score < best2) {
					best2 = tile_best_score;
					best_index2 = tile_best_index;
				} else if (i == 3 && tile_best_score < best3) {
					best3 = tile_best_score;
					best_index3 = tile_best_index;
				}
			}
		}
		__syncthreads();
	}

	if (tx == 0) {
		int p1_start = p1_base + ty * 4;
		score[p1_start + 0] = best0;
		index[p1_start + 0] = best_index0;
		score[p1_start + 1] = best1;
		index[p1_start + 1] = best_index1;
		score[p1_start + 2] = best2;
		index[p1_start + 2] = best_index2;
		score[p1_start + 3] = best3;
		index[p1_start + 3] = best_index3;
	}
}

__global__ void ComputeTop1FromDotV9(const float* dots, const float* lhs_norm, const float* rhs_norm, float* score, int* index, int WIDTH) {
	int row = blockIdx.x;
	int lane = threadIdx.x;
	if (row >= WIDTH) {
		return;
	}

	float best_score = 1e30f;
	int best_index = -1;
	float base_a = lhs_norm[row];

	for (int col = lane; col < WIDTH; col += 32) {
		float d = base_a + rhs_norm[col] + dots[col * WIDTH + row];
		if (d < best_score) {
			best_score = d;
			best_index = col;
		}
	}
	WarpReduceMin(best_score, best_index);
	if (lane == 0) {
		if (best_score < 0.0f && best_score > -1e-4f) {
			best_score = 0.0f;
		}
		score[row] = best_score;
		index[row] = best_index;
	}
}

__global__ void ComputeNearestNeighborV9WMMA(const __half* pts1, const __half* pts2, const float* lhs_norm, const float* rhs_norm, float* score, int* index, int WIDTH) {
	using namespace nvcuda;
	int lane = threadIdx.x & 31;
	int warp_id = threadIdx.x >> 5;
	int a_base = blockIdx.x * 16;
	int row = lane;

	__shared__ float dot_tile[4 * 16 * 16];
	__shared__ float warp_best_score[4 * 16];
	__shared__ int warp_best_index[4 * 16];
	float best_score = 1e30f;
	int best_index = -1;
	float lhs_norm_value = 0.0f;
	if (row < 16 && a_base + row < WIDTH) {
		lhs_norm_value = lhs_norm[a_base + row];
	}

	for (int b_base = 0; b_base < WIDTH; b_base += 64) {
		int b_sub_base = b_base + warp_id * 16;
		wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
		wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
		wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
		wmma::fill_fragment(acc_frag, 0.0f);

		if (b_sub_base < WIDTH) {
			for (int k = 0; k < 128; k += 16) {
				const __half* a_ptr = pts1 + a_base * 128 + k;
				const __half* b_ptr = pts2 + b_sub_base * 128 + k;
				wmma::load_matrix_sync(a_frag, a_ptr, 128);
				wmma::load_matrix_sync(b_frag, b_ptr, 128);
				wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
			}
		}

		wmma::store_matrix_sync(dot_tile + warp_id * 16 * 16, acc_frag, 16, wmma::mem_row_major);
		__syncthreads();

		if (row < 16 && a_base + row < WIDTH && b_sub_base < WIDTH) {
			const float* warp_dot = dot_tile + warp_id * 16 * 16;
			for (int col = 0; col < 16 && b_sub_base + col < WIDTH; col++) {
				float d = lhs_norm_value + rhs_norm[b_sub_base + col] - 2.0f * warp_dot[row * 16 + col];
				if (d < best_score) {
					best_score = d;
					best_index = b_sub_base + col;
				}
			}
		}
		__syncthreads();
	}

	if (row < 16) {
		warp_best_score[warp_id * 16 + row] = best_score;
		warp_best_index[warp_id * 16 + row] = best_index;
	}
	__syncthreads();

	if (warp_id == 0 && row < 16 && a_base + row < WIDTH) {
		float final_best_score = warp_best_score[row];
		int final_best_index = warp_best_index[row];
		for (int i = 1; i < 4; i++) {
			float candidate_score = warp_best_score[i * 16 + row];
			int candidate_index = warp_best_index[i * 16 + row];
			if (candidate_score < final_best_score) {
				final_best_score = candidate_score;
				final_best_index = candidate_index;
			}
		}
		if (final_best_score < 0.0f && final_best_score > -1e-4f) {
			final_best_score = 0.0f;
		}
		score[a_base + row] = final_best_score;
		index[a_base + row] = final_best_index;
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

bool MatchV1(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result) {
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
	if (distance_matrix) cudaFree(distance_matrix);
	if (device_score) cudaFree(device_score);
	if (device_index) cudaFree(device_index);
	if (host_index) cudaFreeHost(host_index);
	if (!success) match_result.clear();
	return success;
}

bool MatchV2(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result) {

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

bool MatchV3(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result) {

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

	ComputeNearestNeighborV3<<<block, thread>>>(lhs_descriptor_device, rhs_descriptor_device, device_score, device_index, size);
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

bool MatchV4(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result) {

	size_t size = lhs.size();
	if (size == 0 || size != rhs.size() || size % 8 != 0) {
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
	dim3 thread(256, 1);
	dim3 block(size / 8, 1);
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

	ComputeNearestNeighborV4<<<block, thread>>>(lhs_descriptor_device, rhs_descriptor_device, device_score, device_index, size);
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

template <typename KernelFn>
bool MatchV5WithKernel(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result, KernelFn kernel) {

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

	kernel<<<block, thread>>>(lhs_descriptor_device, rhs_descriptor_device, device_score, device_index, size);
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

bool MatchV5a(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result) {
	return MatchV5WithKernel(lhs, rhs, match_result, ComputeNearestNeighborV5a);
}

bool MatchV5b(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result) {
	return MatchV5WithKernel(lhs, rhs, match_result, ComputeNearestNeighborV5b);
}

bool MatchV5c(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result) {
	return MatchV5WithKernel(lhs, rhs, match_result, ComputeNearestNeighborV5c);
}

bool MatchV5(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result) {
	return MatchV5c(lhs, rhs, match_result);
}

bool MatchV6(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result) {
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

	ComputeNearestNeighborV6<<<block, thread>>>(lhs_descriptor_device, rhs_descriptor_device, device_score, device_index, size);
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

bool MatchV7(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result) {
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

	ComputeNearestNeighborV7<<<block, thread>>>(lhs_descriptor_device, rhs_descriptor_device, device_score, device_index, size);
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

bool MatchV8(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result) {
	size_t size = lhs.size();
	if (size == 0 || size != rhs.size() || size % 32 != 0) {
		fprintf(stderr, "ERROR Checked : invalid descriptor sizes\n");
		return false;
	}
	match_result.clear();

	__half* lhs_descriptor_host = nullptr;
	__half* rhs_descriptor_host = nullptr;
	__half* lhs_descriptor_device = nullptr;
	__half* rhs_descriptor_device = nullptr;
	float* device_score = nullptr;
	int* host_index = nullptr;
	int* device_index = nullptr;
	dim3 thread(32, 8);
	dim3 block(size / 32, 1);
	size_t offset = 0;
	bool success = false;

	if (!CUDAErrorCheck(cudaMallocHost(&lhs_descriptor_host, sizeof(__half) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMallocHost(&rhs_descriptor_host, sizeof(__half) * size * 128))) goto cleanup;
	for (const Descriptor& descriptor : lhs) {
		for (float value : descriptor) {
			lhs_descriptor_host[offset++] = __float2half_rn(value);
		}
	}
	offset = 0;
	for (const Descriptor& descriptor : rhs) {
		for (float value : descriptor) {
			rhs_descriptor_host[offset++] = __float2half_rn(value);
		}
	}

	if (!CUDAErrorCheck(cudaMalloc(&lhs_descriptor_device, sizeof(__half) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&rhs_descriptor_device, sizeof(__half) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(lhs_descriptor_device, lhs_descriptor_host, sizeof(__half) * size * 128, cudaMemcpyHostToDevice))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(rhs_descriptor_device, rhs_descriptor_host, sizeof(__half) * size * 128, cudaMemcpyHostToDevice))) goto cleanup;

	if (!CUDAErrorCheck(cudaMallocHost(&host_index, sizeof(int) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&device_score, sizeof(float) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&device_index, sizeof(int) * size))) goto cleanup;

	ComputeNearestNeighborV8<<<block, thread>>>(lhs_descriptor_device, rhs_descriptor_device, device_score, device_index, size);
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

bool MatchV9(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result) {
	size_t size = lhs.size();
	if (size == 0 || size != rhs.size() || size % 32 != 0) {
		fprintf(stderr, "ERROR Checked : invalid descriptor sizes\n");
		return false;
	}
	match_result.clear();

	__half* lhs_descriptor_host = nullptr;
	__half* rhs_descriptor_host = nullptr;
	float* lhs_norm_host = nullptr;
	float* rhs_norm_host = nullptr;
	__half* lhs_descriptor_device = nullptr;
	__half* rhs_descriptor_device = nullptr;
	float* lhs_norm_device = nullptr;
	float* rhs_norm_device = nullptr;
	float* device_score = nullptr;
	int* host_index = nullptr;
	int* device_index = nullptr;
	dim3 top1_thread(128);
	dim3 top1_block(size / 16);
	bool success = false;

	if (!CUDAErrorCheck(cudaMallocHost(&lhs_descriptor_host, sizeof(__half) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMallocHost(&rhs_descriptor_host, sizeof(__half) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMallocHost(&lhs_norm_host, sizeof(float) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMallocHost(&rhs_norm_host, sizeof(float) * size))) goto cleanup;
	for (size_t i = 0; i < size; i++) {
		float lhs_norm = 0.0f;
		float rhs_norm = 0.0f;
		for (int k = 0; k < 128; k++) {
			float lhs_value = lhs[i][k];
			float rhs_value = rhs[i][k];
			__half lhs_half = __float2half_rn(lhs_value);
			__half rhs_half = __float2half_rn(rhs_value);
			lhs_descriptor_host[i * 128 + k] = lhs_half;
			rhs_descriptor_host[i * 128 + k] = rhs_half;
			float lhs_value_half = __half2float(lhs_half);
			float rhs_value_half = __half2float(rhs_half);
			lhs_norm += lhs_value_half * lhs_value_half;
			rhs_norm += rhs_value_half * rhs_value_half;
		}
		lhs_norm_host[i] = lhs_norm;
		rhs_norm_host[i] = rhs_norm;
	}

	if (!CUDAErrorCheck(cudaMalloc(&lhs_descriptor_device, sizeof(__half) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&rhs_descriptor_device, sizeof(__half) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&lhs_norm_device, sizeof(float) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&rhs_norm_device, sizeof(float) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(lhs_descriptor_device, lhs_descriptor_host, sizeof(__half) * size * 128, cudaMemcpyHostToDevice))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(rhs_descriptor_device, rhs_descriptor_host, sizeof(__half) * size * 128, cudaMemcpyHostToDevice))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(lhs_norm_device, lhs_norm_host, sizeof(float) * size, cudaMemcpyHostToDevice))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(rhs_norm_device, rhs_norm_host, sizeof(float) * size, cudaMemcpyHostToDevice))) goto cleanup;

	if (!CUDAErrorCheck(cudaMallocHost(&host_index, sizeof(int) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&device_score, sizeof(float) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&device_index, sizeof(int) * size))) goto cleanup;
	ComputeNearestNeighborV9WMMA<<<top1_block, top1_thread>>>(lhs_descriptor_device, rhs_descriptor_device, lhs_norm_device, rhs_norm_device, device_score, device_index, size);
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
	if (lhs_norm_host) cudaFreeHost(lhs_norm_host);
	if (rhs_norm_host) cudaFreeHost(rhs_norm_host);
	if (lhs_descriptor_device) cudaFree(lhs_descriptor_device);
	if (rhs_descriptor_device) cudaFree(rhs_descriptor_device);
	if (lhs_norm_device) cudaFree(lhs_norm_device);
	if (rhs_norm_device) cudaFree(rhs_norm_device);
	if (device_score) cudaFree(device_score);
	if (device_index) cudaFree(device_index);
	if (host_index) cudaFreeHost(host_index);
	if (!success) match_result.clear();
	return success;
}

struct MatchV10Context {
	size_t size;
	__half* lhs_descriptor_device;
	__half* rhs_descriptor_device;
	float* lhs_norm_device;
	float* rhs_norm_device;
	float* device_score;
	int* device_index;
	int* host_index;
};

bool CreateMatchV10Context(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs, MatchV10Context** context) {
	size_t size = lhs.size();
	if (!context || size == 0 || size != rhs.size() || size % 32 != 0) {
		fprintf(stderr, "ERROR Checked : invalid v10 context input\n");
		return false;
	}
	*context = nullptr;

	MatchV10Context* ctx = nullptr;
	__half* lhs_descriptor_host = nullptr;
	__half* rhs_descriptor_host = nullptr;
	float* lhs_norm_host = nullptr;
	float* rhs_norm_host = nullptr;
	bool success = false;

	ctx = new MatchV10Context();
	ctx->size = size;
	ctx->lhs_descriptor_device = nullptr;
	ctx->rhs_descriptor_device = nullptr;
	ctx->lhs_norm_device = nullptr;
	ctx->rhs_norm_device = nullptr;
	ctx->device_score = nullptr;
	ctx->device_index = nullptr;
	ctx->host_index = nullptr;

	if (!CUDAErrorCheck(cudaMallocHost(&lhs_descriptor_host, sizeof(__half) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMallocHost(&rhs_descriptor_host, sizeof(__half) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMallocHost(&lhs_norm_host, sizeof(float) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMallocHost(&rhs_norm_host, sizeof(float) * size))) goto cleanup;
	for (size_t i = 0; i < size; i++) {
		float lhs_norm = 0.0f;
		float rhs_norm = 0.0f;
		for (int k = 0; k < 128; k++) {
			__half lhs_half = __float2half_rn(lhs[i][k]);
			__half rhs_half = __float2half_rn(rhs[i][k]);
			lhs_descriptor_host[i * 128 + k] = lhs_half;
			rhs_descriptor_host[i * 128 + k] = rhs_half;
			float lhs_value = __half2float(lhs_half);
			float rhs_value = __half2float(rhs_half);
			lhs_norm += lhs_value * lhs_value;
			rhs_norm += rhs_value * rhs_value;
		}
		lhs_norm_host[i] = lhs_norm;
		rhs_norm_host[i] = rhs_norm;
	}

	if (!CUDAErrorCheck(cudaMalloc(&ctx->lhs_descriptor_device, sizeof(__half) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&ctx->rhs_descriptor_device, sizeof(__half) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&ctx->lhs_norm_device, sizeof(float) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&ctx->rhs_norm_device, sizeof(float) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&ctx->device_score, sizeof(float) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&ctx->device_index, sizeof(int) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMallocHost(&ctx->host_index, sizeof(int) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(ctx->lhs_descriptor_device, lhs_descriptor_host, sizeof(__half) * size * 128, cudaMemcpyHostToDevice))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(ctx->rhs_descriptor_device, rhs_descriptor_host, sizeof(__half) * size * 128, cudaMemcpyHostToDevice))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(ctx->lhs_norm_device, lhs_norm_host, sizeof(float) * size, cudaMemcpyHostToDevice))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(ctx->rhs_norm_device, rhs_norm_host, sizeof(float) * size, cudaMemcpyHostToDevice))) goto cleanup;

	*context = ctx;
	ctx = nullptr;
	success = true;

cleanup:
	if (lhs_descriptor_host) cudaFreeHost(lhs_descriptor_host);
	if (rhs_descriptor_host) cudaFreeHost(rhs_descriptor_host);
	if (lhs_norm_host) cudaFreeHost(lhs_norm_host);
	if (rhs_norm_host) cudaFreeHost(rhs_norm_host);
	if (ctx) DestroyMatchV10Context(ctx);
	return success;
}

bool RunMatchV10Context(MatchV10Context* context, std::vector<std::pair<int, int>>& match_result) {
	if (!context || context->size == 0) {
		fprintf(stderr, "ERROR Checked : invalid v10 context\n");
		return false;
	}
	match_result.clear();
	dim3 thread(128);
	dim3 block(context->size / 16);
	ComputeNearestNeighborV9WMMA<<<block, thread>>>(context->lhs_descriptor_device, context->rhs_descriptor_device, context->lhs_norm_device, context->rhs_norm_device, context->device_score, context->device_index, context->size);
	if (!CUDAErrorCheck(cudaGetLastError())) {
		return false;
	}
	if (!CUDAErrorCheck(cudaMemcpy(context->host_index, context->device_index, sizeof(int) * context->size, cudaMemcpyDeviceToHost))) {
		return false;
	}
	for (size_t i = 0; i < context->size; i++) {
		match_result.push_back(std::pair<int, int>(static_cast<int>(i), context->host_index[i]));
	}
	return true;
}

void DestroyMatchV10Context(MatchV10Context* context) {
	if (!context) {
		return;
	}
	if (context->lhs_descriptor_device) cudaFree(context->lhs_descriptor_device);
	if (context->rhs_descriptor_device) cudaFree(context->rhs_descriptor_device);
	if (context->lhs_norm_device) cudaFree(context->lhs_norm_device);
	if (context->rhs_norm_device) cudaFree(context->rhs_norm_device);
	if (context->device_score) cudaFree(context->device_score);
	if (context->device_index) cudaFree(context->device_index);
	if (context->host_index) cudaFreeHost(context->host_index);
	delete context;
}

bool MatchV10(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result) {
	MatchV10Context* context = nullptr;
	bool success = false;
	if (!CreateMatchV10Context(lhs, rhs, &context)) {
		match_result.clear();
		return false;
	}
	success = RunMatchV10Context(context, match_result);
	DestroyMatchV10Context(context);
	if (!success) {
		match_result.clear();
	}
	return success;
}

template <typename KernelFn>
bool BenchmarkFloatKernel(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs,const std::vector<int>& expected_match, int warmup_runs, int measured_runs, KernelBenchmarkResult& result, KernelFn kernel, dim3 block, dim3 thread) {
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
		kernel<<<block, thread>>>(lhs_descriptor_device, rhs_descriptor_device, device_score, device_index, size);
	}
	if (!CUDAErrorCheck(cudaGetLastError())) goto cleanup;
	if (!CUDAErrorCheck(cudaDeviceSynchronize())) goto cleanup;

	if (!CUDAErrorCheck(cudaEventCreate(&start))) goto cleanup;
	if (!CUDAErrorCheck(cudaEventCreate(&stop))) goto cleanup;
	for (int i = 0; i < measured_runs; i++) {
		float elapsed_ms = 0.0f;
		if (!CUDAErrorCheck(cudaEventRecord(start, 0))) goto cleanup;
		kernel<<<block, thread>>>(lhs_descriptor_device, rhs_descriptor_device, device_score, device_index, size);
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

bool BenchmarkKernelV1(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs,const std::vector<int>& expected_match, int warmup_runs, int measured_runs, KernelBenchmarkResult& result) {
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

bool BenchmarkKernelV2(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs,const std::vector<int>& expected_match, int warmup_runs, int measured_runs, KernelBenchmarkResult& result) {
	return BenchmarkFloatKernel(lhs, rhs, expected_match, warmup_runs, measured_runs, result, ComputeNearestNeighbor, dim3(lhs.size() / 32, 1), dim3(32, 8));
}

bool BenchmarkKernelV3(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs,const std::vector<int>& expected_match, int warmup_runs, int measured_runs, KernelBenchmarkResult& result) {
	return BenchmarkFloatKernel(lhs, rhs, expected_match, warmup_runs, measured_runs, result, ComputeNearestNeighborV3, dim3(lhs.size() / 32, 1), dim3(32, 8));
}

bool BenchmarkKernelV4(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs,const std::vector<int>& expected_match, int warmup_runs, int measured_runs, KernelBenchmarkResult& result) {
	return BenchmarkFloatKernel(lhs, rhs, expected_match, warmup_runs, measured_runs, result, ComputeNearestNeighborV4, dim3(lhs.size() / 8, 1), dim3(256, 1));
}

bool BenchmarkKernelV5(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs,const std::vector<int>& expected_match, int warmup_runs, int measured_runs, KernelBenchmarkResult& result) {
	return BenchmarkFloatKernel(lhs, rhs, expected_match, warmup_runs, measured_runs, result, ComputeNearestNeighborV5c, dim3(lhs.size() / 32, 1), dim3(32, 8));
}

bool BenchmarkKernelV5a(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs,const std::vector<int>& expected_match, int warmup_runs, int measured_runs, KernelBenchmarkResult& result) {
	return BenchmarkFloatKernel(lhs, rhs, expected_match, warmup_runs, measured_runs, result, ComputeNearestNeighborV5a, dim3(lhs.size() / 32, 1), dim3(32, 8));
}

bool BenchmarkKernelV5b(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs,const std::vector<int>& expected_match, int warmup_runs, int measured_runs, KernelBenchmarkResult& result) {
	return BenchmarkFloatKernel(lhs, rhs, expected_match, warmup_runs, measured_runs, result, ComputeNearestNeighborV5b, dim3(lhs.size() / 32, 1), dim3(32, 8));
}

bool BenchmarkKernelV5c(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs,const std::vector<int>& expected_match, int warmup_runs, int measured_runs, KernelBenchmarkResult& result) {
	return BenchmarkFloatKernel(lhs, rhs, expected_match, warmup_runs, measured_runs, result, ComputeNearestNeighborV5c, dim3(lhs.size() / 32, 1), dim3(32, 8));
}

bool BenchmarkKernelV6(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs,const std::vector<int>& expected_match, int warmup_runs, int measured_runs, KernelBenchmarkResult& result) {
	return BenchmarkFloatKernel(lhs, rhs, expected_match, warmup_runs, measured_runs, result, ComputeNearestNeighborV6, dim3(lhs.size() / 32, 1), dim3(32, 8));
}

bool BenchmarkKernelV7(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs,const std::vector<int>& expected_match, int warmup_runs, int measured_runs, KernelBenchmarkResult& result) {
	return BenchmarkFloatKernel(lhs, rhs, expected_match, warmup_runs, measured_runs, result, ComputeNearestNeighborV7, dim3(lhs.size() / 32, 1), dim3(32, 8));
}

bool BenchmarkKernelV8(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs,const std::vector<int>& expected_match, int warmup_runs, int measured_runs, KernelBenchmarkResult& result) {
	size_t size = lhs.size();
	result.success = false;
	result.best_ms = 0.0f;
	result.avg_ms = 0.0f;
	result.mismatch_count = static_cast<int>(expected_match.size());
	if (size == 0 || size != rhs.size() || size != expected_match.size() || size % 32 != 0 || warmup_runs < 0 || measured_runs <= 0) {
		fprintf(stderr, "ERROR Checked : invalid kernel benchmark input\n");
		return false;
	}

	__half* lhs_descriptor_host = nullptr;
	__half* rhs_descriptor_host = nullptr;
	__half* lhs_descriptor_device = nullptr;
	__half* rhs_descriptor_device = nullptr;
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
	size_t offset = 0;

	if (!CUDAErrorCheck(cudaMallocHost(&lhs_descriptor_host, sizeof(__half) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMallocHost(&rhs_descriptor_host, sizeof(__half) * size * 128))) goto cleanup;
	for (const Descriptor& descriptor : lhs) {
		for (float value : descriptor) {
			lhs_descriptor_host[offset++] = __float2half_rn(value);
		}
	}
	offset = 0;
	for (const Descriptor& descriptor : rhs) {
		for (float value : descriptor) {
			rhs_descriptor_host[offset++] = __float2half_rn(value);
		}
	}

	if (!CUDAErrorCheck(cudaMalloc(&lhs_descriptor_device, sizeof(__half) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&rhs_descriptor_device, sizeof(__half) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&device_score, sizeof(float) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&device_index, sizeof(int) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMallocHost(&host_index, sizeof(int) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(lhs_descriptor_device, lhs_descriptor_host, sizeof(__half) * size * 128, cudaMemcpyHostToDevice))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(rhs_descriptor_device, rhs_descriptor_host, sizeof(__half) * size * 128, cudaMemcpyHostToDevice))) goto cleanup;

	for (int i = 0; i < warmup_runs; i++) {
		ComputeNearestNeighborV8<<<block, thread>>>(lhs_descriptor_device, rhs_descriptor_device, device_score, device_index, size);
	}
	if (!CUDAErrorCheck(cudaGetLastError())) goto cleanup;
	if (!CUDAErrorCheck(cudaDeviceSynchronize())) goto cleanup;

	if (!CUDAErrorCheck(cudaEventCreate(&start))) goto cleanup;
	if (!CUDAErrorCheck(cudaEventCreate(&stop))) goto cleanup;
	for (int i = 0; i < measured_runs; i++) {
		float elapsed_ms = 0.0f;
		if (!CUDAErrorCheck(cudaEventRecord(start, 0))) goto cleanup;
		ComputeNearestNeighborV8<<<block, thread>>>(lhs_descriptor_device, rhs_descriptor_device, device_score, device_index, size);
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

bool BenchmarkKernelV9(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs,const std::vector<int>& expected_match, int warmup_runs, int measured_runs, KernelBenchmarkResult& result) {
	size_t size = lhs.size();
	result.success = false;
	result.best_ms = 0.0f;
	result.avg_ms = 0.0f;
	result.mismatch_count = static_cast<int>(expected_match.size());
	if (size == 0 || size != rhs.size() || size != expected_match.size() || size % 32 != 0 || warmup_runs < 0 || measured_runs <= 0) {
		fprintf(stderr, "ERROR Checked : invalid kernel benchmark input\n");
		return false;
	}

	__half* lhs_descriptor_host = nullptr;
	__half* rhs_descriptor_host = nullptr;
	float* lhs_norm_host = nullptr;
	float* rhs_norm_host = nullptr;
	__half* lhs_descriptor_device = nullptr;
	__half* rhs_descriptor_device = nullptr;
	float* lhs_norm_device = nullptr;
	float* rhs_norm_device = nullptr;
	float* device_score = nullptr;
	int* host_index = nullptr;
	int* device_index = nullptr;
	cudaEvent_t start = nullptr;
	cudaEvent_t stop = nullptr;
	dim3 thread(128);
	dim3 block(size / 16);
	bool success = false;
	float best_ms = 1e30f;
	float sum_ms = 0.0f;

	if (!CUDAErrorCheck(cudaMallocHost(&lhs_descriptor_host, sizeof(__half) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMallocHost(&rhs_descriptor_host, sizeof(__half) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMallocHost(&lhs_norm_host, sizeof(float) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMallocHost(&rhs_norm_host, sizeof(float) * size))) goto cleanup;
	for (size_t i = 0; i < size; i++) {
		float lhs_norm = 0.0f;
		float rhs_norm = 0.0f;
		for (int k = 0; k < 128; k++) {
			__half lhs_half = __float2half_rn(lhs[i][k]);
			__half rhs_half = __float2half_rn(rhs[i][k]);
			lhs_descriptor_host[i * 128 + k] = lhs_half;
			rhs_descriptor_host[i * 128 + k] = rhs_half;
			float lhs_value = __half2float(lhs_half);
			float rhs_value = __half2float(rhs_half);
			lhs_norm += lhs_value * lhs_value;
			rhs_norm += rhs_value * rhs_value;
		}
		lhs_norm_host[i] = lhs_norm;
		rhs_norm_host[i] = rhs_norm;
	}

	if (!CUDAErrorCheck(cudaMalloc(&lhs_descriptor_device, sizeof(__half) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&rhs_descriptor_device, sizeof(__half) * size * 128))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&lhs_norm_device, sizeof(float) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&rhs_norm_device, sizeof(float) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&device_score, sizeof(float) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMalloc(&device_index, sizeof(int) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMallocHost(&host_index, sizeof(int) * size))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(lhs_descriptor_device, lhs_descriptor_host, sizeof(__half) * size * 128, cudaMemcpyHostToDevice))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(rhs_descriptor_device, rhs_descriptor_host, sizeof(__half) * size * 128, cudaMemcpyHostToDevice))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(lhs_norm_device, lhs_norm_host, sizeof(float) * size, cudaMemcpyHostToDevice))) goto cleanup;
	if (!CUDAErrorCheck(cudaMemcpy(rhs_norm_device, rhs_norm_host, sizeof(float) * size, cudaMemcpyHostToDevice))) goto cleanup;

	for (int i = 0; i < warmup_runs; i++) {
		ComputeNearestNeighborV9WMMA<<<block, thread>>>(lhs_descriptor_device, rhs_descriptor_device, lhs_norm_device, rhs_norm_device, device_score, device_index, size);
	}
	if (!CUDAErrorCheck(cudaGetLastError())) goto cleanup;
	if (!CUDAErrorCheck(cudaDeviceSynchronize())) goto cleanup;

	if (!CUDAErrorCheck(cudaEventCreate(&start))) goto cleanup;
	if (!CUDAErrorCheck(cudaEventCreate(&stop))) goto cleanup;
	for (int i = 0; i < measured_runs; i++) {
		float elapsed_ms = 0.0f;
		if (!CUDAErrorCheck(cudaEventRecord(start, 0))) goto cleanup;
		ComputeNearestNeighborV9WMMA<<<block, thread>>>(lhs_descriptor_device, rhs_descriptor_device, lhs_norm_device, rhs_norm_device, device_score, device_index, size);
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
	if (lhs_norm_host) cudaFreeHost(lhs_norm_host);
	if (rhs_norm_host) cudaFreeHost(rhs_norm_host);
	if (lhs_descriptor_device) cudaFree(lhs_descriptor_device);
	if (rhs_descriptor_device) cudaFree(rhs_descriptor_device);
	if (lhs_norm_device) cudaFree(lhs_norm_device);
	if (rhs_norm_device) cudaFree(rhs_norm_device);
	if (device_score) cudaFree(device_score);
	if (device_index) cudaFree(device_index);
	if (host_index) cudaFreeHost(host_index);
	return success;
}

bool BenchmarkKernelV10(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs,const std::vector<int>& expected_match, int warmup_runs, int measured_runs, KernelBenchmarkResult& result) {
	return BenchmarkKernelV9(lhs, rhs, expected_match, warmup_runs, measured_runs, result);
}

bool Match(const std::vector<Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result) {
	return MatchV5(lhs, rhs, match_result);
}
