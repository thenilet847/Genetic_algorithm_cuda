#ifndef GENE_H
#define GENE_H
#include <cuda_runtime.h>
#include <curand_kernel.h>
__device__ __host__ bool get_gene(char* chromosoma, int n_gene); 
__device__ __host__ void set_gene(char* chromosoma, int n_gene, bool value);
__device__ void set_random_genes(curandState* state, char* chromosoma, int size_cromosoma); 
#endif
