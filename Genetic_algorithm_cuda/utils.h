#ifndef UTILS_H
#define UTILS_H
#include <curand_kernel.h>

__global__ void initialize_random_states(curandState* states, unsigned long seed, int population);

__host__ __device__ int get_arr_length_by_genes(int n_genes);

__host__ int* init_precalculated_terms(int genes_size);

__host__ bool malloc_data(curandState** rand_states, char** chromosmas_a, char** chromosmas_b, int** precalc_results, int** results, int n_genes, int population);
__host__ void free_data(curandState* rand_states, char* chromosmas_a, char* chromosmas_b, int* precalc_results, int* results);

__host__ void print_genes(char* chromosoma, int n_genes, int chromosoma_size); 
__host__ void print_all_genes(char* chromosomas, int n_genes, int population, int chromosoma_size); 

#endif
