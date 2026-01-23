
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
#define DEFAULT_POPULATION 5
#define DEFAULT_MAX_GENERATIONS 10000
#define DEFAULT_SELECTION_POOL 5

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


	int gene_arr_size = get_arr_length_by_genes(genes);

    curandState* rand_states; 
    char* crhomosoma_a; 
    char* crhomosoma_b; 
    unsigned int* recalc_results; 
    unsigned int* results; 
    char genes_prova = 0x1; 
    if (malloc_data(&rand_states, &crhomosoma_a, &crhomosoma_b, &recalc_results, &results, genes, population_size)) {
		initialize_random_states << <population_size, 1 >> > (rand_states, time(NULL), population_size);
		generate_random_chromosomas<<<population_size, 1 >>> (rand_states, crhomosoma_a, gene_arr_size, population_size);
		calculate_results << <population_size, 1 >> > (crhomosoma_a, results, recalc_results, gene_arr_size, genes, population_size);
        cudaDeviceSynchronize(); 
        print_all_genes(crhomosoma_a, results, genes, population_size, gene_arr_size);
		perform_mutation << <population_size, 1 >> > (rand_states, crhomosoma_a, gene_arr_size, chromosomas, mutation_rate);
		calculate_results << <population_size, 1 >> > (crhomosoma_a, results, recalc_results, gene_arr_size, genes, population_size);
        cudaDeviceSynchronize(); 
		print_all_genes(crhomosoma_a, results, genes, population_size, gene_arr_size);
		getchar();
    }
	free_data(rand_states, crhomosoma_a, crhomosoma_b, recalc_results, results);
	



    return 0;
}