
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.h"
#include "gene.h"

#include <stdio.h>
#include <time.h>

#define DEFAULT_GENES 40
#define DEFAULT_MUTATION 3 
#define DEFAULT_POPULATION 1000000

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


	int gene_arr_size = get_arr_length_by_genes(genes);

    curandState* rand_states; 
    char* crhomosoma_a; 
    char* crhomosoma_b; 
    int* recalc_results; 
    int* results; 
    if (malloc_data(&rand_states, &crhomosoma_a, &crhomosoma_b, &recalc_results, &results, genes, population_size)) {
		initialize_random_states << <population_size, 1 >> > (rand_states, time(NULL), population_size);
        cudaDeviceSynchronize(); 
        print_all_genes(crhomosoma_a, genes, population_size, gene_arr_size);
		getchar();
    }
	free_data(rand_states, crhomosoma_a, crhomosoma_b, recalc_results, results);
	



    return 0;
}