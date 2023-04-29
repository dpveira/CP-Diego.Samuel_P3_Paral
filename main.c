#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <mpi/mpi.h>

#define DEBUG 0

/* Translation of the DNA bases
   A -> 0
   C -> 1
   G -> 2
   T -> 3
   N -> 4*/

#define M  1000000 // Number of sequences
#define N  200  // Number of bases per sequence

unsigned int g_seed = 0;

int fast_rand(void) {
    g_seed = (214013*g_seed+2531011);
    return (g_seed>>16) % 5;
}

// The distance between two bases
int base_distance(int base1, int base2){

    if((base1 == 4) || (base2 == 4)){
        return 3;
    }

    if(base1 == base2) {
        return 0;
    }

    if((base1 == 0) && (base2 == 3)) {
        return 1;
    }

    if((base2 == 0) && (base1 == 3)) {
        return 1;
    }

    if((base1 == 1) && (base2 == 2)) {
        return 1;
    }

    if((base2 == 2) && (base1 == 1)) {
        return 1;
    }

    return 2;
}

int main(int argc, char *argv[] ) {
    MPI_Init(&argc, &argv);
    int i, j;
    int *data1, *data2, *data3, *data4;
    int *result;
    struct timeval  tv1, tv2;
    int rank, numprocs;

    int tamBloque = ceil(M/numprocs);
    int tiempoTotal = 0;


    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    data1 = (int *) malloc(M * N * sizeof(int));
    data2 = (int *) malloc(M * N * sizeof(int));
    data3 = (int *) malloc(M * N * sizeof(int));
    data4 = (int *) malloc(M * N * sizeof(int));

    result = (int *) malloc(M*sizeof(int));


    if (rank == 0) {
        /* Initialize Matrices */
        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                /* random with 20% gap proportion */
                data1[i * N + j] = fast_rand();
                data2[i * N + j] = fast_rand();
            }
        }
    }

    MPI_Scatter(data1, tamBloque*N, MPI_INT, data3, tamBloque*N, MPI_INT, 0, MPI_COMM_WORLD);//M:10 N:3 NP:2
    MPI_Scatter(data2, tamBloque*N, MPI_INT, data4, tamBloque*N, MPI_INT, 0, MPI_COMM_WORLD);
    gettimeofday(&tv1, NULL);

    if(rank !=0){
        //Primero se calcula para todos los procesos menos el 0
        for(i=0;i<tamBloque;i++) {
            result[i]=0;
            for(j=0;j<N;j++) {
                result[i] += base_distance(data3[i*N+j], data4[i*N+j]);
            }
        }
        //Una vez calculado se envia al proceso 0 para que lo guarde en el array result
        MPI_Gather(result, tamBloque, MPI_INT, NULL, tamBloque, MPI_INT, 0, MPI_COMM_WORLD);
    }else{
        //Se calcula para el proceso 0, como data1 y data2 ya estan inicializados, se calcula directamente
        for(i=0;i<tamBloque;i++) {
            result[i]=0;
            for(j=0;j<N;j++) {
                result[i] += base_distance(data1[i*N+j], data2[i*N+j]);
            }
        }
        //Se recibe el resultado de los demas procesos y se guarda en el array result
        MPI_Gather(NULL, tamBloque, MPI_INT, result, tamBloque, MPI_INT, 0, MPI_COMM_WORLD);
    }
    /*
    for(i=0;i<M;i++) {
        result[i]=0;
        for(j=0;j<N;j++) {
            result[i] += base_distance(data1[i*N+j], data2[i*N+j]);
        }
    }*/

    gettimeofday(&tv2, NULL);

    int microseconds = (tv2.tv_usec - tv1.tv_usec)+ 1000000 * (tv2.tv_sec - tv1.tv_sec);
    //Se calcula el tiempo total de ejecucion de todos los procesos


    MPI_Reduce(&microseconds, &tiempoTotal, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    //Se imprime el tiempo total de ejecucion, se tiene que esperar a que todos los procesos terminen
    if(rank == 0){
        printf("Tiempo total: %d\n", tiempoTotal/numprocs);
    }

    /* Display result */
    if (DEBUG == 1) {
        int checksum = 0;
        for(i=0;i<M;i++) {
            checksum += result[i];
        }
        printf("Checksum: %d\n ", checksum);
    } else if (DEBUG == 2) {
        for(i=0;i<M;i++) {
            printf(" %d \t ",result[i]);
        }
    } else {
        printf ("Time (seconds) = %lf\n", (double) microseconds/1E6);
    }

    free(data1); free(data2); free(result);
    free(data3); free(data4);
    MPI_Finalize();

    return 0;
}