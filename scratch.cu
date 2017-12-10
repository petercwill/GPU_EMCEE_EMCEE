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

#define N_WALKERS 10000
#define MAX_THETA_SIZE 255
#define PARAMS 2
__constant__ float center_d[PARAMS];

// Function declarations
__global__ void init(unsigned int seed, curandState_t *state);
__global__ void init_walkers(float *walkers, int n_walkers, int n_theta, int r, curandState_t *state);
__device__ float G(float a, curandState_t state);
__device__ float Rosenbrock(float *point);
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

__device__ float Rosenbrock(float *point)
{
	return ((float) exp(-((100*pow(point[1] - pow(point[0],2), 2)) + pow(1 - point[0],2)) / 20));
}

__device__ void step_walkers(float *s1_walkers, unsigned int s1_n, float *s2_walkers, unsigned int s2_n, float a, unsigned int k_dim, curandState_t *states)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curandState_t localState = states[id];
	float u1 = curand_uniform(&localState); 
	float u2 = curand_uniform(&localState); 
	float u3 = curand_uniform(&localState); 
    	int id2 = ceil(s2_n*u1)-1;
	float w1[MAX_THETA_SIZE], x_prime[MAX_THETA_SIZE], q1, q2;

	float z = G(a, u2);
	for (int i = 0; i < k_dim; i++)
	{
		x_prime[i] = s2_walkers[id2+(i*s2_n)] + z*(
				s1_walkers[id+(i*s1_n)] - 
				s2_walkers[id2+(i*s2_n)]
				);
	}
    	q1 = Rosenbrock(w1);
	q2 = Rosenbrock(x_prime);
	if (u3 < (powf(z,k_dim-1)*(q2/q1)))
	{
		for (int i =0; i < k_dim; i++)
		{
			s1_walkers[id+(i*s1_n)] = x_prime[i]; 
		}
	}
	states[id] = localState;
		
}

__global__ void emcee_emcee(float *s1_walkers, float *s2_walkers, int k, int k2, unsigned int n_walkers, unsigned int theta_dim, unsigned int steps, int a, curandState_t *states)
{

    	int id = threadIdx.x + blockIdx.x * blockDim.x;

	for(int t = 0; t < steps; t++)
	{
		if (id < k)
		{
			step_walkers(s1_walkers, k, s2_walkers, k2, a, theta_dim, states);	
		}
		__syncthreads();
		if (id < k2)
		{
			step_walkers(s2_walkers, k2, s1_walkers, k, a, theta_dim, states);
		}
		__syncthreads();
	}

}

int get_block_size(int n_walkers)
{
	int factor = n_walkers / 4800;
	int blocksize = factor*32;
	if (blocksize > 256)
	{
		blocksize = 256;
	}	
	return blocksize;
}


int main(int argc, char *argv[]) {

	curandState_t *states;
	int seed = 10;
	int blocksize = get_block_size(N_WALKERS);
	int n_blocks = ceil((float) N_WALKERS / blocksize);
	int a = 2;
	int k = floor((double) N_WALKERS / 2);
	int k2 = N_WALKERS - k;
	float walkers_h_1[PARAMS][k];
	float walkers_h_2[PARAMS][k2];
	float *walkers_d_1, *walkers_d_2;
  	float *center_h = (float*) calloc(PARAMS, sizeof(float)); 
	int r = 2;

	cudaMemcpyToSymbol(center_d, center_h, PARAMS*sizeof(float));
	cudaMalloc((void**) &states, N_WALKERS*sizeof(curandState_t));
	init<<<n_blocks, blocksize>>>(seed, N_WALKERS, states);
	cudaMalloc((void**) &walkers_d_1, .5*PARAMS*N_WALKERS*sizeof(float));
	cudaMalloc((void**) &walkers_d_2, .5*PARAMS*N_WALKERS*sizeof(float));

	init_walkers<<<n_blocks,blocksize>>>(walkers_d_1, k, PARAMS, r, states);
	init_walkers<<<n_blocks,blocksize>>>(walkers_d_2, k2, PARAMS, r, states);
	cudaMemcpy(walkers_h_1, walkers_d_1, PARAMS*k*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(walkers_h_2, walkers_d_2, PARAMS*k2*sizeof(float), cudaMemcpyDeviceToHost);


  	FILE *fp2 = fopen("out.txt", "w");
	for(int i =0; i < k; i++) {
		for(int j=0; j<PARAMS; j++){
			fprintf(fp2, "%f\t", walkers_h_1[j][i]);
		}
		fprintf(fp2, "\n");
	}
	for(int i =0; i < k2; i++) {
		for(int j=0; j<PARAMS; j++){
			fprintf(fp2, "%f\t", walkers_h_2[j][i]);
		}
		fprintf(fp2, "\n");
	}
 	fclose(fp2); 

	emcee_emcee<<<n_blocks,blocksize>>>(walkers_d_1, walkers_d_2, k, k2, N_WALKERS, PARAMS, 10000, a, states);


	cudaMemcpy(walkers_h_1, walkers_d_1, PARAMS*k*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(walkers_h_2, walkers_d_2, PARAMS*k2*sizeof(float), cudaMemcpyDeviceToHost);

  	FILE *fp3 = fopen("out2.txt", "w");
	for(int i =0; i < k; i++) {
		for(int j=0; j<PARAMS; j++){
			fprintf(fp3, "%f\t", walkers_h_1[j][i]);
		}
		fprintf(fp3, "\n");
	}
	for(int i =0; i < k2; i++) {
		for(int j=0; j<PARAMS; j++){
			fprintf(fp3, "%f\t", walkers_h_2[j][i]);
		}
		fprintf(fp3, "\n");
	}
 	fclose(fp3); 
  	return 0;
}
