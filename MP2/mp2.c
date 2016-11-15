#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <xmmintrin.h>

void naive_transpose(int,float*,float*);
void vector_transpose(int,float*,float*);

void naive_matmul(int,int,float*,float*,float*,float*);
void tiled_matmul(int,int,float*, float*,float*,float*);
void matmul1(int,int,float*,float*,float*);
void naive_stencil( int,float*,float*);
void tiled_stencil( int,int,float*,float*);
void stencil(int,int,float*);

void print_out_mat(char*, int,float*,char*); 

int main( int argc, char *argv[]) {
	int s;  // tile size
	char *data_set;   // set1 or set2
	int na,nb,ne; // No of rows or cols in matrix
	if (argc !=3) {
		printf("Usage:%s data_set tile_size\n",argv[0]);
		return 0;
	}
	data_set = argv[1];
	s=atoi(argv[2]);

	char matrix[40];
	FILE *fp1,*fp2,*fp3;
        int i,j;
	
	// reading A,B,E
	sprintf(matrix,"%s_A.txt",data_set);
	fp1=fopen(matrix,"r");
	
	sprintf(matrix,"%s_B.txt",data_set);
	fp2=fopen(matrix,"r");

	sprintf(matrix,"%s_E.txt",data_set);
	fp3=fopen(matrix,"r");
	
	fscanf(fp1,"%d\n",&na);
	fscanf(fp2,"%d\n",&nb);
	fscanf(fp3,"%d\n",&ne);
	float *A = (float*)malloc(na*na*sizeof(float) );
	float *B = (float*)malloc(nb*nb*sizeof(float) );
	float *E = (float*)malloc(ne*ne*sizeof(float) );
        float *E1=(float*)malloc(ne*ne*sizeof(float));
	for(i=0;i<na;i++) {
		for(j=0;j<na;j++) {
	           fscanf(fp1,"%f\n",(A+i*na+j));
		}
  	}

	for(i=0;i<nb;i++) {
		for(j=0;j<nb;j++) {
	           fscanf(fp2,"%f\n",(B+i*nb+j));
		}
  	}

	for(i=0;i<ne;i++) {
		for(j=0;j<ne;j++) {
	           fscanf(fp3,"%f\n",(E+i*ne+j));
		   E1[i*ne+j]=E[i*ne+j];
		}
  	}
          
        fclose(fp1);
	fclose(fp2);
	fclose(fp3);
      /*----Part(a): Transpose of A----*/ 
       float *naive_tr_t=(float*)malloc(sizeof(float));
       float  *vector_tr_t=(float*)malloc(sizeof(float));
       float *useless=(float*)malloc(sizeof(float)); 
      //naive transpose 
      naive_transpose(na,A,naive_tr_t);
      print_out_mat(data_set,na,A,"A");
      //again do transpose to obtain original matrix
      naive_transpose(na,A,useless);
      //vectorized transpose
      vector_transpose(na,A,vector_tr_t);
      print_out_mat(data_set,na,A,"A_v");

      /*----Part(b): C=AB-----*/
      //Re-transposing A
       float *naive_mul_t=(float*)malloc(sizeof(float));
       float  *tiles_mul_t=(float*)malloc(sizeof(float));
       naive_transpose(na,A,useless);
      //C matrix
      float *C=(float*)calloc(na*nb,sizeof(float));
      naive_matmul(na,na,A,B,C,naive_mul_t);
      //print set*_out_C.txt
      print_out_mat(data_set,na,C,"C");
      free(C);

      //Tiled multiplication 
      C=(float*)calloc(na*nb,sizeof(float));
      tiled_matmul(na,s,A,B,C,tiles_mul_t);
      //print set*_out_C_t.txt
      print_out_mat(data_set,na,C,"C_t");

      /*---- Part(c): 5-point stencil-----*/
      float *naive_sten_t = (float*)malloc(sizeof(float));
      float *tiles_sten_t = (float*)malloc(sizeof(float));
      naive_stencil(ne,E,naive_sten_t);
      //print set*_out_E.txt
      print_out_mat(data_set,ne,E,"E");
      tiled_stencil(ne,s,E1,tiles_sten_t);
      //print set*_out_E_t.txt
      print_out_mat(data_set,ne,E1,"E_t");

      free(A);
      free(B);
      free(C);
      free(E);

      /*---Printing speedup results---*/
      char filename[40];
      sprintf(filename,"%s_results.txt",data_set);
      FILE *fp;
      fp=fopen(filename,"w");
      fprintf(fp,"Operation\tNaive_Time\tTiled_Time\tSpeedup\n");
      fprintf(fp,"Transpose\t%f\t%f\t%f\n",*naive_tr_t,*vector_tr_t,(*naive_tr_t)/(*vector_tr_t));
      fprintf(fp,"Matmul\t%f\t\t%f\t%f\n",*naive_mul_t,*tiles_mul_t,(*naive_mul_t)/(*tiles_mul_t));
      fprintf(fp,"Stencil\t%f\t\t%f\t%f\n",*naive_sten_t,*tiles_sten_t,(*naive_sten_t)/(*tiles_sten_t));
      fclose(fp);
      
      return 0;
}

/*Function definitions*/
 void naive_transpose(int na,float *A,float *naive_tr_t) {
      float temp;
      struct timeval start, end;
      int i,j;
      gettimeofday(&start,NULL);
      for(i=0;i<na;i++) {
	  for(j=i+1;j<na;j++){
		  temp=A[i+j*na];
		  A[i+j*na]=A[i*na+j];
		  A[i*na+j]=temp;
	  }
      }
      gettimeofday(&end,NULL);
      *naive_tr_t=(1000000*end.tv_sec+end.tv_usec)-(1000000*start.tv_sec+start.tv_usec);
      return;
	      
 }
 void vector_transpose(int na,float *A,float *vector_tr_t) {
      float temp;
      struct timeval start, end;
      int i,j;
      __m128 row1,row2,row3,row4;
      __m128 row5,row6,row7,row8;
      __m128 r1,r2,r3,r4;
      gettimeofday(&start,NULL);
      for(i=0;i<na;i+=4) {
		  //initial diag submatrix
		  row1=_mm_load_ps(A+i*na+i);
		  row2=_mm_load_ps(A+(i+1)*na+i);
		  row3=_mm_load_ps(A+(i+2)*na+i);
		  row4=_mm_load_ps(A+(i+3)*na+i);
		  // intermediate diagonal submatrix
		  r1=_mm_unpacklo_ps(row1,row3);
		  r2=_mm_unpacklo_ps(row2,row4);
		  r3=_mm_unpackhi_ps(row1,row3);
		  r4=_mm_unpackhi_ps(row2,row4);
                  //transposed diagonal submatrix
		  row1=_mm_unpacklo_ps(r1,r2);
		  row2=_mm_unpackhi_ps(r1,r2);
		  row3=_mm_unpacklo_ps(r3,r4);
		  row4=_mm_unpackhi_ps(r3,r4);
		  // storing back in A
	          _mm_store_ps(A+i*na+i,row1);
	          _mm_store_ps(A+(i+1)*na+i,row2);
	          _mm_store_ps(A+(i+2)*na+i,row3);
	          _mm_store_ps(A+(i+3)*na+i,row4);

	  for(j=i+4;j<na;j+=4){
		  //initial upper-diag submatrix
		  row1=_mm_load_ps(A+i*na+j);
		  row2=_mm_load_ps(A+(i+1)*na+j);
		  row3=_mm_load_ps(A+(i+2)*na+j);
		  row4=_mm_load_ps(A+(i+3)*na+j);
                  // initial lower-diag submatrix          
		  row5=_mm_load_ps(A+(j)*na+i);
		  row6=_mm_load_ps(A+(j+1)*na+i);
		  row7=_mm_load_ps(A+(j+2)*na+i);
		  row8=_mm_load_ps(A+(j+3)*na+i);
		  // intermediate first upper-diagonal submatrix
		  r1=_mm_unpacklo_ps(row1,row3);
		  r2=_mm_unpacklo_ps(row2,row4);
		  r3=_mm_unpackhi_ps(row1,row3);
		  r4=_mm_unpackhi_ps(row2,row4);
                  //transposed first upper-diagonal submatrix
		  row1=_mm_unpacklo_ps(r1,r2);
		  row2=_mm_unpackhi_ps(r1,r2);
		  row3=_mm_unpacklo_ps(r3,r4);
		  row4=_mm_unpackhi_ps(r3,r4);
		  // intermediate lower-diagonal submatrix
		  r1=_mm_unpacklo_ps(row5,row7);
		  r2=_mm_unpacklo_ps(row6,row8);
		  r3=_mm_unpackhi_ps(row5,row7);
		  r4=_mm_unpackhi_ps(row6,row8);
                  //transposed lower-diagonal submatrix
		  row5=_mm_unpacklo_ps(r1,r2);
		  row6=_mm_unpackhi_ps(r1,r2);
		  row7=_mm_unpacklo_ps(r3,r4);
		  row8=_mm_unpackhi_ps(r3,r4);
		  // storing back in A
	          _mm_store_ps(A+i*na+j,row5);
	          _mm_store_ps(A+(i+1)*na+j,row6);
	          _mm_store_ps(A+(i+2)*na+j,row7);
	          _mm_store_ps(A+(i+3)*na+j,row8);
	          _mm_store_ps(A+(j)*na+i,row1);
	          _mm_store_ps(A+(j+1)*na+i,row2);
	          _mm_store_ps(A+(j+2)*na+i,row3);
	          _mm_store_ps(A+(j+3)*na+i,row4);
	  }
      }
      gettimeofday(&end,NULL);
      *vector_tr_t=(1000000*end.tv_sec+end.tv_usec)-(1000000*start.tv_sec+start.tv_usec);
      return;	      
 }

void naive_matmul(int n,int s,float *A,float *B,float *C,float *naive_mul_t){
	int i,j,k;
	struct timeval start,end;

	gettimeofday(&start,NULL);
	for(i=0;i<s;i++){
	   for(k=0;k<s;k++){
	      for(j=0;j<s;j++){
	         C[i*n+j]=C[i*n+j]+A[i*n+k]*B[k*n+j];
	      }
	   }
	}
	gettimeofday(&end,NULL);
	*naive_mul_t=(1000000*end.tv_sec + end.tv_usec)-(1000000*start.tv_sec + start.tv_usec);
}
void tiled_matmul(int n,int s,float *A,float *B,float *C,float *tiles_mul_t){
	int i,j,k;
	struct timeval start,end;

	gettimeofday(&start,NULL);
	for(i=0;i<n;i+=s){
	   for(k=0;k<n;k+=s){
	      for(j=0;j<n;j+=s){
	         matmul1(n,s,(A+i*n+k),(B+k*n+j),(C+i*n+j));
	      }
	   }
	}
	gettimeofday(&end,NULL);
	*tiles_mul_t=(1000000*end.tv_sec + end.tv_usec)-(1000000*start.tv_sec + start.tv_usec);
}
void matmul1(int n,int s,float *A,float *B,float *C){
	int i,j,k;
	for(i=0;i<s;i++){
	   for(k=0;k<s;k++){
	      for(j=0;j<s;j++){
	         C[i*n+j]=C[i*n+j]+A[i*n+k]*B[k*n+j];
	      }
	   }
	}
}

void naive_stencil(int ne,float *E,float *naive_sten_t){
	int i,j;
	struct timeval start,end;
	
	gettimeofday(&start,NULL);
	for(i=1;i<(ne-1);i++){
           for(j=1;j<(ne-1);j++){
               E[i*ne+j]=2e-1*(E[i*ne+j]+(E[i*ne+j-1]+E[i*ne+j+1])+(E[(i-1)*ne+j]+E[(i+1)*ne+j]));
	   }
	}
	gettimeofday(&end,NULL);
	*naive_sten_t=(1000000*end.tv_sec + end.tv_usec)-(1000000*start.tv_sec + start.tv_usec);
}
void tiled_stencil(int ne,int s,float *E,float *tile_sten_t){
	int i,j;
	struct timeval start,end;
	
	gettimeofday(&start,NULL);
	for(i=1;i<(ne-1);i+=s){
           for(j=1;j<(ne-1);j+=s){
	      stencil(ne,s,(E+i*ne+j));
	   }
	}
	gettimeofday(&end,NULL);
	*tile_sten_t=(1000000*end.tv_sec + end.tv_usec)-(1000000*start.tv_sec + start.tv_usec);
}

void stencil(int ne,int s,float *E){
        int i,j;	
	for(i=0;i<s;i++){
           for(j=0;j<s;j++){
               E[i*ne+j]=2e-1*(E[i*ne+j]+(E[i*ne+j-1]+E[i*ne+j+1])+(E[(i-1)*ne+j]+E[(i+1)*ne+j]));
	   }
	}
}
void print_out_mat(char *data_set,int na,float *A,char *suff){
	char filename[40];
	sprintf(filename,"%s_out_%s.txt",data_set,suff);
	FILE *fp;
	fp=fopen(filename,"w");
	int i,j;

	for(i=0;i<na;i++){
           for(j=0;j<na;j++){
		   fprintf(fp,"%1.2f\n",*(A+i*na+j));
	   }
	}
	fclose(fp);
	return;
}

