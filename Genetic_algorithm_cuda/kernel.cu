#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.h"
#include "gene.h"
#include "phases.h"
#include <curand_kernel.h>

#include <stdio.h>
#include <time.h>

#define DEFAULT_GENES 40
#define DEFAULT_MUTATION 3 
#define DEFAULT_POPULATION 200
#define DEFAULT_MAX_GENERATIONS 10000
#define DEFAULT_SELECTION_POOL 5
#define DEFAULT_SEARCH_NUMBER 4097

void get_print_device_info() {
	int nDevices;
	cudaGetDeviceCount(&nDevices);

	printf("Number of devices: %d\n", nDevices);

	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("  Total global memory (Gbytes) %.1f\n", (float)(prop.totalGlobalMem) / 1024.0 / 1024.0 / 1024.0);
		printf("  Shared memory per block (Kbytes) %.1f\n", (float)(prop.sharedMemPerBlock) / 1024.0);
		printf("  minor-major: %d-%d\n", prop.minor, prop.major);
		printf("  Warp-size: %d\n", prop.warpSize);
		printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
	}
}



int main()
{
	get_print_device_info();




	int population_size = DEFAULT_POPULATION;
	int genes = DEFAULT_GENES;
	int chromosomas = DEFAULT_POPULATION;
	int mutation_rate = DEFAULT_MUTATION;
	int search_number = DEFAULT_SEARCH_NUMBER; 
	int selection_pool = DEFAULT_SELECTION_POOL;


	int gene_arr_size = get_arr_length_by_genes(genes);

	curandState* rand_states = NULL;
	char* chromosoma_a = NULL;
	char* chromosoma_h = NULL;
	unsigned int* selected_index = NULL;
	unsigned int* selected_index_h = NULL;
	unsigned int* recalc_results = NULL;
	unsigned int* results = NULL;

	if (malloc_data(&rand_states, &chromosoma_a, &chromosoma_h, &selected_index, &selected_index_h, &recalc_results, &results, genes, population_size)) {
		initialize_random_states << <population_size, 1 >> > (rand_states, time(NULL), population_size);
		generate_random_chromosomas << <population_size, 1 >> > (rand_states, chromosoma_a, gene_arr_size, population_size);
		cudaMemcpyAsync(chromosoma_h, chromosoma_a, sizeof(char) * chromosomas * gene_arr_size, cudaMemcpyDeviceToHost);
		calculate_results << <population_size, 1 >> > (chromosoma_a, results, recalc_results, gene_arr_size, genes, population_size);
		cudaDeviceSynchronize();
		selection<<<population_size, 1>>>(rand_states, chromosoma_a, selected_index, recalc_results, results, population_size, selection_pool, gene_arr_size);
		cudaDeviceSynchronize();
		cudaMemcpy(selected_index_h, selected_index, sizeof(unsigned int) * population_size, cudaMemcpyDeviceToHost);
		/*for (int i = 0; i < chromosomas; i++) {
			printf("selected index for %d is %d\n", i, selected_index_h[i]); 
		}*/
		//print_all_genes(chromosoma_a, results, genes, population_size, gene_arr_size);
		apply_selection(chromosoma_a, chromosoma_h, selected_index_h, gene_arr_size, population_size);
		cudaDeviceSynchronize();
		//perform_mutation << <population_size, 1 >> > (rand_states, chromosoma_a, gene_arr_size, chromosomas, mutation_rate);
		calculate_results << <population_size, 1 >> > (chromosoma_a, results, recalc_results, gene_arr_size, genes, population_size);
		cudaDeviceSynchronize();
		//print_all_genes(chromosoma_a, results, genes, population_size, gene_arr_size);
		getchar();
	}
	else {
		fprintf(stderr, "Failed to allocate data on device and host.\n");
	}
	free_data(rand_states, chromosoma_a, chromosoma_h, selected_index, selected_index_h, recalc_results, results);


	return 0;
}