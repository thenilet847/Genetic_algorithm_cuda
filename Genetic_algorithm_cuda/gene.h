#ifndef GENE_H
#define GENE_H
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
__device__ __host__ bool get_gene(char* chromosoma, int n_gene); 
__device__ __host__ void set_gene(char* chromosoma, int n_gene, bool value);
__device__ void set_random_genes(curandState* state, char* chromosoma, int size_cromosoma); 
__device__ unsigned int get_chromosoma_result(char* chromosoma, unsigned int* precalc_results, int chromosoma_length, int n_genes);
__device__ void mutate_chromosoma(curandState* state, char* chromosoma, int chromosoma_size, int mutation_rate);
#endif
