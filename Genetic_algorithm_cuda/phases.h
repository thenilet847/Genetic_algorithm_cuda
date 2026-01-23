#ifndef PHASES_H
#define PHASES_H
#include <cuda_runtime.h>
#include <curand_kernel.h>
__global__ void generate_random_chromosomas(curandState* states, char* chromosomas, int chromosoma_size, int population);
__global__ void calculate_results(char* chromosomas, unsigned int* results, unsigned int* precalc_results, int chromosoma_size, int n_genes, int population);
__global__ void perform_mutation(curandState* states, char* chromosomas, int chromosoma_size, int population, int mutation_rate);
#endif