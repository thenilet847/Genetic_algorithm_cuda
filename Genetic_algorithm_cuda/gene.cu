#include "gene.h"
#include <device_launch_parameters.h>
#include <stdio.h>
/// <summary>
/// Returns boolean value of the gene at position n_gene in the chromosoma
/// </summary>
/// <param name="chromosoma">Chromosoma array</param>
/// <param name="n_gene">Position at gene to get</param>
/// <returns>Boolean value of gene state</returns>
__device__ __host__ bool get_gene(char* chromosoma, int n_gene) {
	char gene = chromosoma[n_gene >> 3]; 
	char filter = 1 << (n_gene & 0x7);
	return (gene & filter); 
}
/// <summary>
/// Set value of the gene at position n_gene in the chromosoma
/// </summary>
/// <param name="chromosoma">Chromosoma to set value</param>
/// <param name="n_gene">Position of value</param>
/// <param name="value">New value of gene</param>
__device__ __host__ void set_gene(char* chromosoma, int n_gene, bool value) {
	char gene = chromosoma[n_gene >> 3];
	char filter = 1 << (n_gene & 0x7);
	if (value) {
		gene |= filter; 
	}
	else {
		gene &= ~filter; 
	}
	chromosoma[n_gene >> 3] = gene;
}
/// <summary>
/// Method to set random genes in a chromosoma
/// </summary>
/// <param name="states">curandState to calculate random values</param>
/// <param name="chromosoma">chromosoma array</param>
/// <param name="size_cromosoma">size of array of genes</param>
/// <returns></returns>
__device__ void set_random_genes(curandState* state, char* chromosoma, int size_cromosoma) {
	for (int i = 0; i < size_cromosoma; i++) {
		chromosoma[i] = (char)(curand(state));
	}
}