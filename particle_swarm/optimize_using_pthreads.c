/* Implementation of PSO using pthreads.
 *
 * Author: Naga Kandasamy
 * Date: January 31, 2025
 *
 */
#define _GNU_SOURCE
#define BILLION 1000000000L  // For nanosecond conversion

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "pso.h"
#include <pthread.h>

/* Data structure defining what to pass to each worker thread */
typedef struct thread_data_s {
    int tid;                        /* The thread ID */
    int num_threads;                /* Number of threads in the worker pool */
    int num_elements;               /* Number of elements in the vector */
    swarm_t *swarm;                /* Pointer to swarm of particles */
    float w;                    
    float c1;
    float c2;
    float xmin;
    float xmax;
    char * function;        /* function to optimize*/
    float curr_fitness;
    float local_min_best_fit;
    int local_min_best_g;  
    int offset;                     /* Starting offset for each thread within the vectors */ 
    int chunk_size;                 /* Chunk size */
} thread_data_t;
void *optimize_swarm(void *args);

int g = -1; 
int best_fitness = INFINITY;
pthread_barrier_t barrier;
pthread_mutex_t mutex;

//The vector x and vector y need to be changes into PSOs

 int optimize_using_pthreads(char *function, int dim, int swarm_size, 
                             float xmin, float xmax, int num_iter, int num_threads)
 {
 
    struct timespec start, end;
    double elapsed_time;

    // Start timing
    clock_gettime(CLOCK_MONOTONIC, &start);

     /* Initialize PSO */
     swarm_t *swarm;
     srand(time(NULL));
     swarm = pso_init(function, dim, swarm_size, xmin, xmax);
 
     int i, j, iter, local_min_best_g;
     float w, c1, c2,local_best_fitness;
     
     float curr_fitness;
 
     w = 0.79;
     c1 = 1.49;
     c2 = 1.49;
     iter = 0;
     local_min_best_g = -1;
     local_best_fitness=INFINITY;
 
 
 
     pthread_t *thread_id = (pthread_t *) malloc (num_threads * sizeof(pthread_t)); /* Data structure to store the thread IDs */
     pthread_attr_t attributes;      /* Thread attributes */
     pthread_attr_init(&attributes); /* Initialize thread attributes to default values */
     pthread_barrier_init(&barrier, NULL, num_threads);
     pthread_mutex_init(&mutex, NULL);
     particle_t *particle;
 
  
     int chunk_size = (int)floor((float) swarm->num_particles/(float) num_threads); /* Compute the chunk size */
     
     thread_data_t *thread_data = (thread_data_t *) malloc(sizeof(thread_data_t) * num_threads);	  
     for (int i = 0; i < num_threads; i++) {
         // i = 0 , 1 ,2 .. 
         // off = 0, 12 ,24... 
         thread_data[i].tid = i; 
         thread_data[i].num_threads = num_threads;
         thread_data[i].num_elements = swarm->num_particles; 
         thread_data[i].swarm = swarm;
         thread_data[i].w  = w; 
         thread_data[i].c1 = c1;
         thread_data[i].c2 = c2;
         thread_data[i].xmin = xmin;
         thread_data[i].xmax = xmax;
         thread_data[i].function = function; 
         thread_data[i].curr_fitness = curr_fitness;
         thread_data[i].local_min_best_fit = local_best_fitness;
         thread_data[i].local_min_best_g = local_min_best_g;
         thread_data[i].offset = i * chunk_size; 
         thread_data[i].chunk_size = chunk_size;
     }

 
     while (iter < num_iter) {
 
 
         //optimize threads till here 
     /* Fork point: allocate memory on heap for required data structures and create worker threads */
 
 
 
         for (int i = 0; i < num_threads; i++)
             pthread_create(&thread_id[i], &attributes, optimize_swarm, (void *)&thread_data[i]);
                         
         for (int i = 0; i < num_threads; i++)
             pthread_join(thread_id[i], NULL);
 
 
 
         /* Identify best performing particle. HINT: You can inline this function when optimizing using pthreads. */
         // g = pso_get_best_fitness(swarm);
         // for (int i = 0; i < swarm->num_particles; i++) {
         //     particle = &swarm->particle[i];
         //     particle->g = g;
         // }
 
         iter++;
     } /* End of iteration */
     // Stop timing
     clock_gettime(CLOCK_MONOTONIC, &end);

     // Compute elapsed time
     elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / (double)BILLION;

     pthread_barrier_destroy(&barrier);
     pthread_mutex_destroy(&mutex);
     free((void *)thread_data);
     
     printf("Execution time: %.6f seconds\n", elapsed_time);
     printf("Our particle Solution:\n");
     pso_print_particle(&swarm->particle[g]);
 
     return g;
 }

void *optimize_swarm(void *args)
{

    /* Typecast argument as a pointer to the thread_data_t structure */
    thread_data_t *thread_data = (thread_data_t *)args; 	

    unsigned int seed = time(NULL); // Seed the random number generator 
    /* 
1: swarm pointer
2: float w
3: float c1 
4: float c2
5: float xmin
6: float xmax
7: char * function 
8: float curr_fitness
*/
    float r1, r2;
    int i,j; 
    /* Compute the partial sum that this thread is responsible for. */
    particle_t *particle, *gbest;

    if (thread_data->tid < (thread_data->num_threads - 1)) {        
        for (i = 0; i < thread_data->offset; i++) {
            particle = &thread_data->swarm->particle[i];
            gbest = &thread_data->swarm->particle[particle->g];  /* Best performing particle from last iteration */ 
            for (j = 0; j < particle->dim; j++) {   /* Update this particle's state */
                r1 = (float)rand_r(&seed)/(float)RAND_MAX;
                r2 = (float)rand_r(&seed)/(float)RAND_MAX;
                /* Update particle velocity */
                particle->v[j] = thread_data->w * particle->v[j]\
                                 + thread_data->c1 * r1 * (particle->pbest[j] - particle->x[j])\
                                 + thread_data->c2 * r2 * (gbest->x[j] - particle->x[j]);
                /* Clamp velocity */
                if ((particle->v[j] < -fabsf(thread_data->xmax - thread_data->xmin)) || (particle->v[j] > fabsf(thread_data->xmax - thread_data->xmin))) 
                    particle->v[j] = uniform(-fabsf(thread_data->xmax - thread_data->xmin), fabsf(thread_data->xmax - thread_data->xmin));

                /* Update particle position */
                particle->x[j] = particle->x[j] + particle->v[j];
                if (particle->x[j] > thread_data->xmax)
                    particle->x[j] = thread_data->xmax;
                if (particle->x[j] < thread_data->xmin)
                    particle->x[j] = thread_data->xmin;
            } /* State update */

            /* Evaluate current fitness */
            pso_eval_fitness(thread_data->function, particle, &thread_data->curr_fitness);

            /* Update pbest */
            if (thread_data->curr_fitness < particle->fitness) {
                particle->fitness = thread_data->curr_fitness;
                for (j = 0; j < particle->dim; j++)
                    particle->pbest[j] = particle->x[j];
            }
        } /* Particle loop */
        pthread_barrier_wait(&barrier);
        for (int i = 0; i < thread_data->offset; i++) {
            particle = &thread_data->swarm->particle[i];
            if (particle->fitness < thread_data->local_min_best_fit) {
                thread_data->local_min_best_fit = particle->fitness;
                thread_data->local_min_best_g = i;
            }
        }
        pthread_mutex_lock(&mutex);  
        if (thread_data->local_min_best_fit < best_fitness)
        {
            best_fitness = thread_data->local_min_best_fit;
            g = thread_data->local_min_best_g; 
        }
        pthread_mutex_unlock(&mutex);  

        pthread_barrier_wait(&barrier);

        for (int i = 0; i < thread_data->offset; i++) {
            particle = &thread_data->swarm->particle[i];
            particle->g = g;
        }
    } 
    else { /* This takes care of the number of elements that the final thread must process */
        for (i = 0; i < thread_data->num_elements; i++) {
            particle = &thread_data->swarm->particle[i];
            gbest = &thread_data->swarm->particle[particle->g];  /* Best performing particle from last iteration */ 
            for (j = 0; j < particle->dim; j++) {   /* Update this particle's state */
                r1 = (float)rand_r(&seed)/(float)RAND_MAX;
                r2 = (float)rand_r(&seed)/(float)RAND_MAX;
                /* Update particle velocity */
                particle->v[j] = thread_data->w * particle->v[j]\
                                 + thread_data->c1 * r1 * (particle->pbest[j] - particle->x[j])\
                                 + thread_data->c2 * r2 * (gbest->x[j] - particle->x[j]);
                /* Clamp velocity */
                if ((particle->v[j] < -fabsf(thread_data->xmax - thread_data->xmin)) || (particle->v[j] > fabsf(thread_data->xmax - thread_data->xmin))) 
                    particle->v[j] = uniform(-fabsf(thread_data->xmax - thread_data->xmin), fabsf(thread_data->xmax - thread_data->xmin));

                /* Update particle position */
                particle->x[j] = particle->x[j] + particle->v[j];
                if (particle->x[j] > thread_data->xmax)
                    particle->x[j] = thread_data->xmax;
                if (particle->x[j] < thread_data->xmin)
                    particle->x[j] = thread_data->xmin;
            } /* State update */

            /* Evaluate current fitness */
            pso_eval_fitness(thread_data->function, particle, &thread_data->curr_fitness);

            /* Update pbest */
            if (thread_data->curr_fitness < particle->fitness) {
                particle->fitness = thread_data->curr_fitness;
                for (j = 0; j < particle->dim; j++)
                    particle->pbest[j] = particle->x[j];
            }
        } /* Particle loop */
        pthread_barrier_wait(&barrier);

        for (int i = 0; i < thread_data->num_elements; i++) {
            particle = &thread_data->swarm->particle[i];
            if (particle->fitness < thread_data->local_min_best_fit) {
                thread_data->local_min_best_fit = particle->fitness;
                thread_data->local_min_best_g = i;
            }
        }

        pthread_mutex_lock(&mutex);
        if (thread_data->local_min_best_fit < best_fitness)
        {

            best_fitness = thread_data->local_min_best_fit;
            //printf("thread_data->local_min_best_fit: %d\n", best_fitness);

            //printf("thread_data->local_min_best_g: %d\n", thread_data->local_min_best_g);
            //error is here
            g = thread_data->local_min_best_g; 

        }
        pthread_mutex_unlock(&mutex);  

        pthread_barrier_wait(&barrier);

        for (int i = 0; i < thread_data->num_elements; i++) {
            particle = &thread_data->swarm->particle[i];
            particle->g = g;
        }

    }

    pthread_exit(NULL);
}
