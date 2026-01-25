#include "phases.h"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include "gene.h"

__global__ void generate_random_chromosomas(curandState* states, char* chromosomas, int chromosoma_size, int population) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= population) return;
	set_random_genes(&states[idx], &chromosomas[idx * chromosoma_size], chromosoma_size);
}
__global__ void calculate_results(char* chromosomas, unsigned int* results, unsigned int* precalc_results, int chromosoma_size, int n_genes, int population) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	if (idx >= population) return; 
	results[idx] = get_chromosoma_result(&chromosomas[idx * chromosoma_size], precalc_results, chromosoma_size, n_genes);
}
__global__ void perform_mutation(curandState* states, char* chromosomas, int chromosoma_size, int population, int mutation_rate) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= population) return;
	mutate_chromosoma(&states[idx], &chromosomas[idx * chromosoma_size], chromosoma_size, mutation_rate);
}
__global__ void selection(curandState* states, char* device_chromosomas, unsigned int* selected_index, unsigned int* precalc_device, unsigned int* results, unsigned int population, unsigned int selection_pool, int chromosoma_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= population) return;
	unsigned int act_pos, best_pos, act_result, best_result;
	best_pos = curand(&states[idx]) % population;
	best_result = results[best_pos];
	for (unsigned int i = 1; i < selection_pool; i++) {
		act_pos = curand(&states[idx]) % population;
		act_result = results[act_pos];
		if (act_result > best_result) {
			best_result = act_result;
			best_pos = act_pos;
		}
	}
	selected_index[idx] = best_pos;
}
__host__ void apply_selection(char* device_chromosomas, char* host_chromosomas, unsigned int* selected_index_h, int chromosoma_size, int population) {
	for (int i = 0; i < population; i++) {
		cudaMemcpyAsync((void*)(&device_chromosomas[i*chromosoma_size]), (void*)(&host_chromosomas[selected_index_h[i]*chromosoma_size]), chromosoma_size * sizeof(char), cudaMemcpyHostToDevice);
	}
}