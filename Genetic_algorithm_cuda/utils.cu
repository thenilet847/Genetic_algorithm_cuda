#include "utils.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "global_defines.h"
#include <stdio.h>
#include <curand_kernel.h>
#include <curand.h>

/// <summary>
/// Initialize random states for each thread
/// </summary>
/// <param name="states">Array of states</param>
/// <param name="seed">Seed for random states</param>
/// <param name="population">Number of population</param>
__global__ void initialize_random_states(curandState* states, unsigned long seed, int population) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < population) {
		curand_init(seed, idx, 0, &states[idx]);
	}
}
/// <summary>
/// Calculate length of char array to store genes
/// </summary>
/// <param name="genes_size">Number of genes</param>
/// <returns>Length of genes array</returns>
__host__ __device__ int get_arr_length_by_genes(int genes_size) {
	int size = genes_size >> 3; 
	if (genes_size & 0x7) size++; 
	return size; 
}
/// <summary>
/// Initialize precalculated terms on device
/// </summary>
/// <param name="genes_size">Genes count</param>
/// <returns>Return reference of precalculated terms array</returns>
__host__ unsigned int* init_precalculated_terms(int n_genes) {
	unsigned int* prec_h; 
	prec_h = (unsigned int*)malloc(sizeof(unsigned int) * n_genes);
	if (prec_h == NULL) return NULL; 
	for (int i = 1; i <= n_genes; i++) {
		prec_h[i-1] = i * i; 
	}
	unsigned int* prec_d; 
	cudaError_t err = cudaMalloc((void**)&prec_d, sizeof(int) * n_genes);
	if (err == cudaSuccess) cudaMemcpy((void*)prec_d, (void*)prec_h, sizeof(unsigned int) * n_genes, cudaMemcpyHostToDevice);
	free(prec_h); 
	if (err != cudaSuccess) return NULL;
	return prec_d;
}
/// <summary>
/// Malloc and precalculate data on device
/// </summary>
/// <param name="rand_states">reference to reference to rand_states</param>
/// <param name="chromosmas_a">reference to reference to chromosomas_a</param>
/// <param name="chromosmas_b">reference to reference to chromosomas_b</param>
/// <param name="precalc_results">reference to reference to precalc_results</param>
/// <param name="results">reference to reference to results</param>
/// <param name="n_genes">Number of genes</param>
/// <param name="population">Number of population</param>
/// <returns>True if success, false otherwise</returns>
__host__ bool malloc_data(curandState** rand_states, char** chromosmas_a, char** chromosoma_h, unsigned int** selected_index, unsigned int** selected_index_h, unsigned int** precalc_results, unsigned int** results, int n_genes, int population) {
	int chromo_size = get_arr_length_by_genes(n_genes);
	cudaError_t err;
	err = cudaMalloc((void**)rand_states, sizeof(curandState) * population);
	if (err != cudaSuccess) return false;
	err = cudaMalloc((void**)chromosmas_a, sizeof(char) * chromo_size * population);
	if (err != cudaSuccess) return false;
	err = cudaMalloc((void**)selected_index, sizeof(unsigned int) * population);
	if (err != cudaSuccess) return false;
	err = cudaMalloc((void**)results, sizeof(unsigned int) * population);
	if (err != cudaSuccess) return false;
	*precalc_results = init_precalculated_terms(n_genes);
	if (*precalc_results == NULL) return false;
	err = cudaMallocHost(chromosoma_h, sizeof(char) * chromo_size * population); 
	if (err != cudaSuccess) return false; 
	err = cudaMallocHost((void**)selected_index_h, sizeof(unsigned int) * population);
	return (err == cudaSuccess);
}
/// <summary>
/// Free memory from device
/// </summary>
/// <param name="rand_states">pointer to rand_states</param>
/// <param name="chromosmas_a">pointer to chromosomas_a</param>
/// <param name="chromosmas_b">pointer to chromosomas_b</param>
/// <param name="precalc_results">pointer to precalc_results</param>
/// <param name="results">pointer to results</param>
__host__ void free_data(curandState* rand_states, char* chromosmas_a, char* chromosoma_h, unsigned int* selected_index, unsigned int* selected_index_host, unsigned int* precalc_results, unsigned int* results) {
	if (rand_states != NULL) {
		cudaFree(rand_states);
	}
	if (chromosmas_a != NULL) {
		cudaFree(chromosmas_a);
	}
	if (selected_index != NULL) {
		cudaFree(selected_index);
	}
	if (precalc_results != NULL) {
		cudaFree(precalc_results);
	}
	if (results != NULL) {
		cudaFree(results);
	}
	if (chromosoma_h != NULL && chromosoma_h != (char*) - 1l) {
		cudaFreeHost(chromosoma_h);
	}
	if (selected_index_host != NULL && selected_index_host != (unsigned int*) -1l) {
		cudaFreeHost(selected_index_host);
	}
}
__host__ void print_genes(char* chromosoma, int n_genes, int chromosoma_size) {
	int i, j;
	char act_part;
	char* resp;
	resp = (char*)malloc(sizeof(char) * (n_genes + 1));
	if (resp == NULL) {
		fprintf(stderr, "Can not allocate memory at RAM.\n");
		return;
	}
	resp[n_genes] = 0;
	for (i = 0; i < chromosoma_size; i++) {
		act_part = chromosoma[i];
		for (j = 0; j < 8; j++) {
			if ((i << 3) + j >= n_genes) break;
			resp[(i << 3) + j] = (act_part & 0x1 ? '1' : '0');
			act_part >>= 1;
		}
	}
	printf("%s\n", resp);
	free(resp);
}
__host__ void print_all_genes(char* chromosomas, unsigned int* results, int n_genes, int population, int chromosoma_size) {
	char* chromosoma = (char*)malloc(chromosoma_size); 
	unsigned int result; 
	if (chromosoma == NULL) {
		fprintf(stderr, "Can not allocate memory at RAM. \n"); 
		return; 
	}
	for (int i = 0; i < population; i++) {
		cudaMemcpy((void*)chromosoma, &chromosomas[i * chromosoma_size], sizeof(char) * chromosoma_size, cudaMemcpyDeviceToHost); 
		cudaMemcpy((void*)&result, (void*)&results[i], sizeof(int), cudaMemcpyDeviceToHost);
		printf("Chromosoma %d\t(value=%u): \t", i, result);
		print_genes(chromosoma, n_genes, chromosoma_size); 
	}
	free(chromosoma); 
}