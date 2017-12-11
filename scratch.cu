/*
*
* Last name: Will
* First name: Peter
* Net ID: pcw276
*
*/

#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define N_WALKERS 1000 
#define MAX_THETA_SIZE 255
#define PARAMS 2
__constant__ float center_d[PARAMS];

// Function declarations
__global__ void init(unsigned int seed, curandState_t *state);
__global__ void init_walkers(float *walkers, int n_walkers, int n_theta, int r, curandState_t *state);
__device__ float G(float a, curandState_t state);
__device__ double Rosenbrock(float *point);
__device__ void step_walkers(float *s1_walkers, unsigned int s1_n, float *s2_walkers, unsigned int s2_n, float a, unsigned int k_dim, curandState_t *states);
__global__ void emcee_emcee(float *walkers, unsigned int n_walkers, unsigned int theta_dim, unsigned int steps, int a, curandState_t *states);
/********************************************/
__global__ void init(unsigned int seed, unsigned int n_walkers, curandState_t *state)
{
    int id = threadIdx.x + blockIdx.x *blockDim.x;
    if (id < n_walkers)
    {
	    curand_init(seed, id, 0, &state[id]);
    }
}

__global__ void init_walkers(float *walkers, int n_walkers, int n_theta, int r, curandState_t *state)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id < n_walkers)
	{
		for(int i = 0; i < n_theta; i++)
		{
			walkers[(id+(i*n_walkers))] = center_d[i] +
				(curand_uniform(&state[id])-.5)*2*r;
		}
	}
}

__device__ float G(float a, float u)
{
	return pow((u*(a-1)+1) / sqrtf(a), 2);
}

__device__ double Rosenbrock(float *point)
{
	return ((double) exp(-((100*pow(point[1] - pow(point[0],2), 2)) + pow(1 - point[0],2)) / 20));
}

__device__ void step_walkers(float *walkers, unsigned int n_walkers, unsigned int k, unsigned int offset, unsigned int theta_dim, float a, curandState_t *states)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curandState_t localState = states[id];
	float u1 = curand_uniform(&localState); 
	float u2 = curand_uniform(&localState); 
	float u3 = curand_uniform(&localState); 

	int w1_idx = id + offset;
    	int w2_idx = k+ceil((n_walkers - k)*u1)-1 - offset;
	float w1[MAX_THETA_SIZE], x_prime[MAX_THETA_SIZE];
        double q1, q2;


	float z = G(a, u2);
	for (int i = 0; i < theta_dim; i++)
	{
		w1[i] = walkers[w1_idx+(i*n_walkers)];
		x_prime[i] = walkers[w2_idx+(i*n_walkers)] + z*(
				walkers[w1_idx+(i*n_walkers)] - 
				walkers[w2_idx+(i*n_walkers)]
				);
	}
    	q1 = Rosenbrock(w1);
	q2 = Rosenbrock(x_prime);
	if (u3 < (powf(z,theta_dim-1)*(q2/q1)))
	{
		for (int i =0; i < theta_dim; i++)
		{
			walkers[w1_idx+(i*n_walkers)] = x_prime[i]; 
		}
	}
	states[id] = localState;
		
}

__global__ void emcee_emcee(float *walkers, int k, int k2, unsigned int n_walkers, unsigned int theta_dim, int a, curandState_t *states)
{

    	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id < k)
	{
		step_walkers(walkers, n_walkers, k, 0, a, theta_dim, states);	
	}
	__syncthreads();

	if (id < k2)
	{
		step_walkers(walkers, n_walkers, k2, k, a, theta_dim, states);
	}

}

int get_block_size(int n_walkers)
{
	int factor = ceil((double)n_walkers / 4800);
	int blocksize = factor*32;
	if (blocksize > 256)
	{
		blocksize = 256;
	}	
	return blocksize;
}

void get_mean(float *walkers, double *means,
		unsigned int n_walkers,
	       	unsigned int theta_dim, int step)
{
	float *start_ind, *stop_ind;

	for(int i = 0; i < theta_dim; i++)
	{
		start_ind = walkers + i*n_walkers;
		stop_ind = walkers + (i+1)*n_walkers;
		thrust::device_vector<float> vec(
				start_ind, stop_ind
				);
		means[i + theta_dim*step] = thrust::reduce(
				vec.begin(), vec.end()
				) / n_walkers;
	}


}


int main(int argc, char *argv[]) {

	curandState_t *states;
	int seed = 10;
	int blocksize = get_block_size(N_WALKERS);
	int n_blocks = ceil((float) N_WALKERS / blocksize);
	int a = 2;
	int k = floor((double) N_WALKERS / 2);
	int k2 = N_WALKERS - k;
	float *walkers_h, *walkers_d;
  	float *center_h = (float*) calloc(PARAMS, sizeof(float)); 
	int r = 2;

	walkers_h = (float*) malloc(PARAMS*N_WALKERS*sizeof(float));
	cudaMemcpyToSymbol(center_d, center_h, PARAMS*sizeof(float));
	cudaMalloc((void**) &states, N_WALKERS*sizeof(curandState_t));
	init<<<n_blocks, blocksize>>>(seed, N_WALKERS, states);
	cudaMalloc((void**) &walkers_d, PARAMS*N_WALKERS*sizeof(float));
	init_walkers<<<n_blocks*2,blocksize>>>(walkers_d, N_WALKERS, PARAMS, r, states);

	printf("USING %d blocks of %d threads\n", 
			n_blocks, blocksize);

	cudaMemcpy(walkers_h, walkers_d, PARAMS*N_WALKERS*sizeof(float), cudaMemcpyDeviceToHost);

  	FILE *fp2 = fopen("out.txt", "w");
	for(int i =0; i < N_WALKERS; i++) {
		for(int j=0; j<PARAMS; j++){
			fprintf(fp2, "%f\t", walkers_h[i +j*N_WALKERS]);
		}
		fprintf(fp2, "\n");
	}
 	fclose(fp2);

	int STEPS = 10000;
	double *means;
	means = (double *) malloc(PARAMS*STEPS*sizeof(double));
	for(int t = 0; t<STEPS; t++)
	{	

		emcee_emcee<<<n_blocks,blocksize>>>(walkers_d, 
				k, k2, N_WALKERS, PARAMS,
			       	a, states);


  		get_mean(walkers_d, means, N_WALKERS, PARAMS, t);
	}
	FILE *fp4 = fopen("means.txt","w");
	for(int j =0; j<STEPS; j++)
	{
		for(int i =0; i<PARAMS; i++)
		{
			fprintf(fp4, "%f \t", means[i +j*PARAMS]);
		}
			fprintf(fp4, "\n");
	}
	fclose(fp4);

	cudaMemcpy(walkers_h, walkers_d, PARAMS*N_WALKERS*sizeof(float), cudaMemcpyDeviceToHost);

  	FILE *fp3 = fopen("out2.txt", "w");
	for(int i =0; i < N_WALKERS; i++) {
		for(int j=0; j<PARAMS; j++){
			fprintf(fp3, "%f\t", walkers_h[i+j*N_WALKERS]);
		}
		fprintf(fp3, "\n");
	}
 	fclose(fp3); 
  	return 0;
}
