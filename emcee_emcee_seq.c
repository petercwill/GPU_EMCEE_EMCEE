#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/**************************************************************/
#define MAX_THETA_SIZE 256
#define Index(w, t, n) ((w) + ((t)*(n)))

/**************************************************************/
void init_walkers(float *walkers, unsigned int n_walkers, 
	unsigned int theta_n, float r);

double Rosenbrock(float *walker);

float G(float a, float u);

void ensemble_mean(float *walkers, int n_walkers, int n_theta,
	       	int step, double *means);

void step_walkers(float *walkers, int n_walkers, int n_theta,
	       	int k, int offset, float a);

void emcee_emcee(float *walkers, int n_walkers, int n_theta,
		unsigned int steps, float a, double *means);

void means_to_file(double* means, unsigned int steps, 
		unsigned int n_theta, const char *f_name);
/**************************************************************/
int main(int argc, char *argv[])
{
	float *walkers;
	float r = 2.0;
	int seed = 10;
	float a = 2.0;
	double *means;

	if(argc != 4)
	{
		fprintf(stderr, "usage: emcee_emcee_seq" 
				"n_walkers, n_theta, steps\n");

		fprintf(stderr, "n_walkers: number of walkers"
			       	"to use in the ensemble\n");

		fprintf(stderr, "n_theta: the dimension of the"
			       	"probability space to sample"
			       	"from\n");

		fprintf(stderr, "steps = number of steps each"
			       	"walker will take in the"
			       	"simulation");
		exit(1);
	}

	unsigned int n_walkers = atoi(argv[1]);
	unsigned int n_theta = atoi(argv[2]);
	unsigned int steps = atoi(argv[3]);
		
	srand(seed);	
	walkers = malloc(n_walkers*n_theta*sizeof(float));
	means = malloc(n_theta*steps*sizeof(double));

	init_walkers(walkers, n_walkers, n_theta, r);
	emcee_emcee(walkers, n_walkers, n_theta, steps, a, means);
	const char *f_means = "means_seq.out";
	means_to_file(means, steps, n_theta, f_means); 

}

/**************************************************************/
void init_walkers(float *walkers, unsigned int n_walkers, 
	unsigned int theta_n, float r)
{
	for(int w = 0; w < n_walkers; w++)
	{
		for(int t = 0; t < theta_n; t++)
		{
			walkers[Index(w, t, n_walkers)] =
		       	((float) rand()/(float)(RAND_MAX/r));
		}
	}
}

/**************************************************************/

double Rosenbrock(float *walker)
{
	return ((double) exp(-((100*pow(walker[1] - pow(walker[0],2), 2)) + pow(1 - walker[0], 2)) / 20));
}


/**************************************************************/
float G(float a, float u)
{
	return pow((u*(a-1)+1) / sqrtf(a),2);
}


/**************************************************************/
void ensemble_mean(float *walkers, int n_walkers, int n_theta, int step, double *means)
{
	double mean;
	for(int t=0; t<n_theta; t++)
	{
		mean = 0;
		for(int w=0; w<n_walkers; w++)
		{
			mean += walkers[Index(w,t,n_walkers)];
		}
		means[t + (step*n_theta)] = mean / n_walkers;
	}
}


/**************************************************************/
void step_walkers(float *walkers, int n_walkers, int n_theta,
	       	int k, int offset, float a)
{
	int w2_idx;
	double q1, q2;
	float w1[MAX_THETA_SIZE], x_prime[MAX_THETA_SIZE], z;
	float u1 = (float) rand() / (float) RAND_MAX;
	float u2 = (float) rand() / (float) RAND_MAX;

	for(int w1_idx=offset; w1_idx<k+offset; w1_idx++)
	{
		w2_idx = k + (rand() % (n_walkers - k)) - offset;

		z = G(a, u1);	
		for(int t=0; t<n_theta; t++)
		{
			w1[t] = walkers[Index(w1_idx, t, n_walkers)];
			x_prime[t] =
			       	walkers[Index(w2_idx, t, n_walkers)]
			       	+ z*(w1[t] - walkers[
						Index(w2_idx,t,
						       	n_walkers)]);

		}	
		q1 = Rosenbrock(w1);
		q2 = Rosenbrock(x_prime);
		if(u2 < (powf(z, n_theta - 1)*(q2/q1)))
		{
			for(int t=0; t < n_theta; t++)
			{
				walkers[Index(w1_idx, t,
					       	n_walkers)] = 
					x_prime[t];
			}	
		}
	}
}	


/**************************************************************/
void emcee_emcee(float *walkers, int n_walkers, int n_theta,
		unsigned int steps, float a, double *means)
{
	int k = floor((double) n_walkers / 2);
	int k2 = n_walkers - k;

	for(int s=0; s<steps; s++)
	{
		step_walkers(walkers, n_walkers, n_theta,
			       	k, 0, a); 
		step_walkers(walkers, n_walkers, n_theta,
				k2, k, a);
		ensemble_mean(walkers, n_walkers, n_theta,
			       	s, means);
	}
}


/**************************************************************/
void means_to_file(double* means, unsigned int steps, 
		unsigned int n_theta, const char *f_name)
{
	FILE *fp = fopen(f_name, "w");
	for(int s = 0; s < steps; s++)
	{
		for(int t = 0; t < n_theta; t++)
		{
			fprintf(fp, "%f\t", means[t + n_theta*s]);
		}
		fprintf(fp, "\n");
	}
}

