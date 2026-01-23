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