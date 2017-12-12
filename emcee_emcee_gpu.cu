#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>


#define IDX(w, t, n_walkers) ((w) + ((t)*(n_walkers)))
/***************************************************************/
__global__ void init_curand_states(int seed,
	       	unsigned int size, curandState_t *states);
	
__global__ void init_walkers(float *walkers, unsigned int
	       	n_walkers, unsigned int n_theta, unsigned int r,
		curandState_t *states);

__global__ void step_walkers(float *walkers, unsigned int 
		n_walkers, unsigned int s1_sz, unsigned int 
		offset, unsigned int n_theta, float a,
		curandState_t *states);

void walkers_to_file(float* walkers, unsigned int n_walkers, 
		unsigned int n_theta, const char *f_name);
/***************************************************************/
unsigned int get_block_size(unsigned int n_walkers)
{
	unsigned int factor = ceil((double)n_walkers / 4800);
	unsigned int blocksize = factor*32;
	if(blocksize > 256)
	{
		blocksize = 256;
	}
	return blocksize;
}


int main(int argc, char*argv[])
{
	curandState_t *states;
	float *walkers_h, *walkers_d;
	
	int seed = 10;
	int a = 2;
	int r = 2;

	if(argc !=4)
	{
		fprintf(stderr, "usage emcee_emcee_gpu "
				"n_walkers, n_theta, steps\n");

		fprintf(stderr, "n_walkers: number of walkers "
				"to use in ensemble\n");

		fprintf(stderr, "n_theta: the dimension of the "
				"probability space to sample "
				"from \n");

		fprintf(stderr, "steps: number of steps each "
				"walker will take in the "
				"simulation\n");
		return 1;

	}

	unsigned int n_walkers = atoi(argv[1]);
	unsigned int n_theta = atoi(argv[2]);
	unsigned int steps = atoi(argv[3]);

	unsigned int s1_sz = ceil((float) n_walkers / 2);
	unsigned int s2_sz = n_walkers - s1_sz;

	unsigned int block_sz = get_block_size(n_walkers);
	unsigned int n_blocks = ceil((float) n_walkers
		       	/ block_sz);

	long states_byte_sz = n_walkers*sizeof(curandState_t);
	long walker_byte_sz = n_walkers*n_theta*sizeof(float);
	unsigned int s_mem_sz = 2*n_theta*sizeof(float);

	fprintf(stdout,"LAUNCHING %d BLOCKS OF SIZE %d\n",
		       	n_blocks, block_sz);

	// allocate and init individual random number seeds
	cudaMalloc((void**) &states, states_byte_sz);
	init_curand_states<<<2*n_blocks,block_sz>>>(
			seed, n_walkers, states);

	// allocate and init each walker.
	walkers_h = (float*) malloc(walker_byte_sz);
	cudaMalloc((void**) &walkers_d, walker_byte_sz);
	init_walkers<<<2*n_blocks,block_sz>>>(walkers_d, n_walkers,
			n_theta, r, states);


	for(int s = 0; s < steps; s++)
	{
		//step with first half of walkers
		step_walkers<<<n_blocks, block_sz, s_mem_sz>>>(
				walkers_d, n_walkers, s1_sz, 0,
			       	n_theta, a, states);

		//step with second half of walkers
		step_walkers<<<n_blocks, block_sz, s_mem_sz>>>(
				walkers_d, n_walkers, s2_sz, s1_sz,
			       	n_theta, a, states);
	}

	cudaMemcpy(walkers_h, walkers_d, walker_byte_sz, cudaMemcpyDeviceToHost);
const char* f_name2 = "test2.out";
walkers_to_file(walkers_h, n_walkers, n_theta, f_name2); 
}
/***************************************************************/
__global__ void init_curand_states(int seed,
	       	unsigned int size, curandState_t *states)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	if(id < size)
	{
		curand_init(seed, id, 0, &states[id]);
	}
}
/***************************************************************/
__global__ void init_walkers(float *walkers, unsigned int
	       	n_walkers, unsigned int n_theta, unsigned int r,
		curandState_t *states)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	if(id < n_walkers)
	{
		for(int t = 0; t < n_theta; t++)
		{
			walkers[IDX(id,t,n_walkers)] =
			       	2*r*(curand_uniform(&states[id])
						-.5);
		}
	}
}

/***************************************************************/
/* this is inverse CDF of the proposal distribution suggested
   in Weare and Goodman 2010.
   
   Parameter a is scaling value nominally set to 2.0.
   
   Parameter u is a random uniform drawn from [0, 1].
   
   The return value is random draw from the proposal
   distribution.
 */

__device__ float G(float a, float u)
{
	return powf((u*(a-1)+1) / sqrtf(a), 2);
}

/***************************************************************/
/* The Rosenbrock distribution is the test distribution we
   wish to approximate expected values from.
   
   See https://en.wikipedia.org/wiki/Rosenbrock_function for
   details.
*/


__device__ double Rosenbrock(float *walker)
{
	return ((double) exp(-((100*pow(walker[1] 
				- pow(walker[0],2),2)) +
				       	pow(1 - walker[0],2))
			       	/ 20)); 
}

/***************************************************************/
__global__ void step_walkers(float *walkers, unsigned int 
		n_walkers, unsigned int s1_sz, unsigned int 
		offset, unsigned int n_theta, float a,
		curandState_t *states)
{

	int id = threadIdx.x + (blockIdx.x * blockDim.x);
	if(id < s1_sz)
	{
		extern __shared__ float w1[];
		float *w1_prime = &w1[n_theta];

		curandState_t localState = states[id];

		int w1_idx = id + offset;
		int w2_idx = s1_sz + ceil((n_walkers - s1_sz) *
			curand_uniform(&localState)) - 
			1 - offset; 
		float z = G(a,curand_uniform(&localState));
		for(int t=0; t<n_theta; t++)
		{
			w1[t] = walkers[IDX(w1_idx, t, n_walkers)];
			w1_prime[t] = walkers[IDX(w2_idx, t, n_walkers)]
				+ z*(w1[t] - walkers[IDX(w2_idx,
						       	t, n_walkers)]);
		}
		if (curand_uniform(&localState) < 
				(powf(z,n_theta-1)*(
					Rosenbrock(w1_prime) / 
					Rosenbrock(w1))
				))
		{
			for(int t=0; t<n_theta; t++)
			{
				walkers[IDX(w1_idx, t, n_walkers)] =
				       	w1_prime[t];
	
			}
		}
		states[id] = localState;
	}
}

/***************************************************************/
void walkers_to_file(float* walkers, unsigned int n_walkers, 
		unsigned int n_theta, const char *f_name)
{
	FILE *fp = fopen(f_name,"w");
	for(int w = 0; w < n_walkers; w++)
	{
		for(int t = 0; t < n_theta; t++){
			fprintf(fp, "%f\t", walkers[
					IDX(w,t,n_walkers)]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}

/***************************************************************/

void get_means(float *walkers, double *means, unsigned int n_walkers,
		unsigned int n_theta, int step)
{
	float *start_ind, *stop_ind;

	for(int t =0; t < n_theta; t++)
	{
		start_ind = walkers
