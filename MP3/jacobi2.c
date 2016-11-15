#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include "mpi.h"

//MPI implementation with Overlap of Computation and Communication
//Constants are being used instead of arguments
#define BC_HOT  1.0
#define BC_COLD 0.0
#define INITIAL_GRID 0.5
#define TOL 1.0e-4

struct timeval tv;
double get_clock() {
   struct timeval tv; int ok;
   ok = gettimeofday(&tv, (void *) 0);
   if (ok<0) { printf("gettimeofday error");  }
   return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}


double **create_matrix(int n) {
	int i;
	double **a;

	a = (double**) malloc(sizeof(double*)*n);
	for (i=0;i<n;i++) {
		a[i] = (double*) malloc(sizeof(double)*n);
	}

	return a;
}

void init_matrix(double **a, int n) {

	int i, j;
	
	for(i=0; i<n; i++) {
		for(j=0; j<n; j++)
			a[i][j] = INITIAL_GRID;
	}
}

void swap_matrix(double ***a, double ***b) {

	double **temp;

	temp = *a;
	*a = *b;
	*b = temp;	
}

void print_grid(double **a, int nstart, int nend) {

	int i, j;

	for(i=nstart; i<nend; i++) {
		for(j=nstart; j<nend; j++) {
			printf("%6.4lf ",a[i][j]);
		}
		printf("\n");
	}
}

void free_matrix(double **a, int n) {
	int i;
	for (i=0;i<n;i++) {
		free(a[i]);
	}
	free(a);
}

int main(int argc, char* argv[]) {
	int i,j,iteration;
	int n = atoi(argv[1]);
        int max_iter = atoi(argv[2]);
        int R = atoi(argv[3]); int C = atoi(argv[4]);
        int R_low, C_low;
        int tile_size, p;
        int size, rank;
        int my_row, my_col;
        int dest,src; 
        int north = 0, south=1, west = 2, east = 3;
	double **a, **b, maxdiff, maxdiff_all;
	double tstart, tend, ttotal;
        double *left_col_send, *right_col_send;
        double *left_col_recv, *right_col_recv;
        MPI_Status status[8];
        MPI_Request request[8]; 
       
        MPI_Init(&argc,&argv);
        MPI_Comm_size(MPI_COMM_WORLD,&size);
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);

        p = size;
        tile_size = n /sqrt(p); 

        //coordinates of processor
        my_row = (rank - rank % (int)sqrt(p)) / (int)sqrt(p);
        my_col = rank % (int)sqrt(p);

        //global starting coordinates of tile
        R_low = my_row * tile_size;
        C_low = my_col * tile_size; 
        
	//add 2 to each dimension for ghost cells
	a = create_matrix(tile_size+2);
	b = create_matrix(tile_size+2);
        left_col_send = (double*) malloc(sizeof(double)*tile_size);
        right_col_send = (double*) malloc(sizeof(double)*tile_size);
        left_col_recv = (double*) malloc(sizeof(double)*tile_size);
        right_col_recv = (double*) malloc(sizeof(double)*tile_size);

        //initialize inner grid in tile
	init_matrix(a,tile_size+2);

	//Initialize the hot boundaries
        //top row
        if (my_row == 0) {
	   for(j=0; j < tile_size+2; j++) {
              a[0][j] = BC_HOT;
	   }
        }
        //left column
        if (my_col == 0) { 
	   for(i=0; i < tile_size+2; i++) {
              a[i][0] = BC_HOT;
	   }
        }
        //right column
        if (my_col == (int)sqrt(p) - 1) { 
	   for(i=0; i < tile_size+2; i++) {
              a[i][tile_size + 1] = BC_HOT;
	   }
        }

	// Initialize the cold boundary
        //bottom row
        if (my_row == (int)sqrt(p) - 1) { 
	   for(j=0; j < tile_size+2; j++) {
              a[tile_size + 1][j] = BC_COLD;
	   }
        }

	// Copy a to b
	for(i=0; i < tile_size+2; i++) {
		for(j=0; j < tile_size+2; j++) {
	        b[i][j] = a[i][j];
		}
	}
	// Main simulation routine
	iteration=0;
	maxdiff_all=1.0;
        if (rank == 0)
	   printf("Running simulation with tolerance=%lf and max iterations=%d\n",
		   TOL, max_iter);
        MPI_Barrier(MPI_COMM_WORLD);
	tstart = get_clock();
	while(maxdiff_all > TOL && iteration < max_iter) {

                maxdiff = 0;
                
                //Send boundary values
                //top row
                if (my_row != 0){
                   dest = sqrt(p) * (my_row-1) + my_col;
                   MPI_Isend(a[1]+1,tile_size,MPI_DOUBLE,dest,north,MPI_COMM_WORLD,request);
                }
                //bottom row
                if (my_row != sqrt(p) - 1){
                   dest = sqrt(p) * (my_row+1) + my_col;
                   MPI_Isend(a[tile_size]+1,tile_size,MPI_DOUBLE,dest,south,MPI_COMM_WORLD,request+1);
                }
                //left column
                if (my_col != 0){
                   //copying left and right columns
                   for(i=0;i<tile_size;i++){
                      left_col_send[i] = a[i+1][1];
                   }
                   dest = sqrt(p) * my_row + my_col - 1;
                   MPI_Isend(left_col_send,tile_size,MPI_DOUBLE,dest,west,MPI_COMM_WORLD,request+2);
                }
                //right column
                if (my_col != sqrt(p) - 1){
                   //copying left and right columns
                   for(i=0;i<tile_size;i++){
                      right_col_send[i] = a[i+1][tile_size];
                    }
                   dest = sqrt(p) * my_row + my_col + 1;
                   MPI_Isend(right_col_send,tile_size,MPI_DOUBLE,dest,east,MPI_COMM_WORLD,request+3);
                }
                   
                //Recieve in ghost cells
                //get bottom row from top row of src
                if (my_row != sqrt(p) - 1){
                   src = sqrt(p) * (my_row + 1) + my_col;
                   MPI_Irecv(a[tile_size+1]+1,tile_size,MPI_DOUBLE,src,north,MPI_COMM_WORLD,request+4);
                }
                //get top row from bottom row of src
                if (my_row != 0){
                   src = sqrt(p) * (my_row - 1) + my_col;
                   MPI_Irecv(a[0]+1,tile_size,MPI_DOUBLE,src,south,MPI_COMM_WORLD,request+5);
                }
                //get right column from left column of src
                if (my_col != sqrt(p) - 1){
                   src = sqrt(p) * my_row + my_col + 1;
                   MPI_Irecv(right_col_recv,tile_size,MPI_DOUBLE,src,west,MPI_COMM_WORLD,request+6);
                }
                //get left column from right column of src
                if (my_col != 0){
                   src = sqrt(p) * my_row + my_col - 1;
                   MPI_Irecv(left_col_recv,tile_size,MPI_DOUBLE,src,east,MPI_COMM_WORLD,request+7);
                }

		// Compute new grid values
		for(i=2; i < tile_size; i++) {
			for(j=2; j < tile_size; j++) {
			   b[i][j] = 0.2 * (a[i][j] + a[i-1][j] + a[i+1][j] + a[i][j-1] + a[i][j+1]);
                           if (fabs(b[i][j] - a[i][j]) > maxdiff)
                               maxdiff = fabs(b[i][j] - a[i][j]);
			}
		}

                if (my_row != sqrt(p) - 1){
                   MPI_Wait(request+4,status);
                }
                //get top row from bottom row of src
                if (my_row != 0){
                   MPI_Wait(request+5,status+1);
                }
                //get right column from left column of src
                if (my_col != sqrt(p) - 1){
                   MPI_Wait(request+6,status+2);
                   for(i=0;i<tile_size;i++){
                      a[i+1][tile_size+1] = right_col_recv[i];
                   }
                }
                //get left column from right column of src
                if (my_col != 0){
                   MPI_Wait(request+7,status+3);
                   for(i=0;i<tile_size;i++){
                      a[i+1][0] = left_col_recv[i];
                   }
                }

 
                //computing real boundaries of tile
                //top and bottom boundary
                for (j=1; j <tile_size +1; j++){
                    b[1][j]  = 0.2 * (a[1][j] + a[0][j] + a[2][j] + a[1][j-1] + a[1][j+1]); 
                    b[tile_size][j] = 0.2 * (a[tile_size][j] + a[tile_size-1][j] + a[tile_size+1][j]
                                           + a[tile_size][j-1] + a[tile_size][j+1]);
                    if (fabs(b[1][j] - a[1][j]) > maxdiff)
                       maxdiff = fabs(b[1][j] - a[1][j]);
                    if (fabs(b[tile_size][j] - a[tile_size][j]) > maxdiff)
                       maxdiff = fabs(b[tile_size][j] - a[tile_size][j]);
                }
                //left and right boundary
                for (i=1;i < tile_size + 1; i++){
                    b[i][1] = 0.2 * (a[i][1] + a[i-1][1] + a[i+1][1] + a[i][0] + a[i][2]);
                    b[i][tile_size] = 0.2 * (a[i][tile_size] + a[i-1][tile_size] + a[i+1][tile_size]
                                           + a[i][tile_size-1] + a[i][tile_size + 1]); 
                    if (fabs(b[i][1] - a[i][1]) > maxdiff)
                       maxdiff = fabs(b[i][1] - a[i][1]);
                    if (fabs(b[i][tile_size] - a[i][tile_size]) > maxdiff)
                       maxdiff = fabs(b[i][tile_size] - a[i][tile_size]);

                }

                //wait for send of bottom and top rows to complete
                //top row
                if (my_row != 0){
                   MPI_Wait(request,status); 
                }
                //bottom row
                if (my_row != sqrt(p) - 1){
                   MPI_Wait(request+1,status+1); 
                }
		// Copy b to a
		swap_matrix(&a,&b);	

                //compute for maxdiff across all processors
                MPI_Allreduce(&maxdiff,&maxdiff_all,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
                
		iteration++;
	}
        MPI_Barrier(MPI_COMM_WORLD);
	tend = get_clock();
	ttotal = tend-tstart;


	// Results
        // Printing value at (R,C)
        if (R_low < R && R < R_low + tile_size && C_low < C && C < C_low + tile_size){
	   printf("Results:\n");
	   printf("Iterations=%d\n",iteration);
	   printf("Tolerance=%12.10lf\n",maxdiff_all);
	   printf("Running time=%12.10lf\n",ttotal);
           printf("Value at R,C=%12.10lf\n",a[R-R_low+1][C-C_low+1]); 
           printf("========================================================================================================\n");
           printf("========================================================================================================\n");
        }

	free_matrix(a,tile_size+2);
	free_matrix(b,tile_size+2);
        //free(left_col_send);
        //free(left_col_recv);
        //free(right_col_send);
        //free(right_col_recv);

        MPI_Finalize();
	return 0;
}
