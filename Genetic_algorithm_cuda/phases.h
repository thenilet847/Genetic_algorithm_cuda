#ifndef PHASES_H
#define PHASES_H
#include <cuda_runtime.h>
#include <curand_kernel.h>
__global__ void generate_random_chromosomas(curandState* states, char* chromosomas, int chromosoma_size, int population);
__global__ void calculate_results(char* chromosomas, unsigned int* results, unsigned int* precalc_results, int chromosoma_size, int n_genes, int population);
__global__ void perform_mutation(curandState* states, char* chromosomas, int chromosoma_size, int population, int mutation_rate);
__global__ void selection(curandState* states, char* device_chromosomas, unsigned int* selected_index, unsigned int* precalc_device, unsigned int* results, unsigned int population, unsigned int selection_pool, int chromosoma_size);
__host__ void apply_selection(char* device_chromosomas, char* host_chromosomas, unsigned int* selected_index, int chromosoma_size, int population); 
#endif