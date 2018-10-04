/*
 ============================================================================
 Name        : EdilioMassaro.c
 Author      : Edilio Massaro
 Description : Progetto MPI esame PcPc
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpi.h"
#define n 5000
#define nc 5000

int main(int argc, char* argv[]) {
	int my_rank;/* rank of process */
	int p; /* number of processes */
	MPI_Status status;
	MPI_Init(&argc, &argv);
	int rowNumber = 0;
	int start = 0;
	int end = 0;
	int work = n - 2;
	double diffnorm;
	double dif;
	int flag = 0;
	int iter = 0;
	double t1, t2; 

	MPI_Comm topology ;
	/* find out process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	/* find out number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	int i, j;

	//Allocazione dinamica delle matrici
	double *x;
	double *xnew;
	xnew = malloc((n*nc)*sizeof(float*));
	x = malloc((n*nc)*sizeof(float*));




	int col = nc-1;
	//inizializzazione della matrice
	for (i = 0; i < n; i++) {
		for (j = 0; j < nc; j++) {
			x[i*nc+j] = rand() % (10 + 1 - 0) + 0;
		}
	}



	//Calcolo del master del il numero di righe per processo da computare e le spedisco tramite comunicazione point to point agli slave
	if (my_rank == 0) {
		rowNumber = work / p;
		printf("RowNumber %d e resto %d \n", rowNumber, (n-2) % p);
		int tmpEnd;
		int tmpStart;
		start = 1;
		//controllo del resto sul master
		if (my_rank < work % p) {
			end = start + rowNumber + 1;
		}
		else{
			end = start + rowNumber;
		}
		tmpEnd = end;
		tmpStart = start;
		//assegnamento degli indici ai processi slave
		for (int i = 1; i < p; i++) {
			tmpStart = tmpEnd;
			//controllo del resto sullo slave
			if (i < work % p) {
				tmpEnd = tmpStart + rowNumber+1;
			}
			else{
				tmpEnd = tmpStart + rowNumber;
			}
			MPI_Send(&tmpStart, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&tmpEnd, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
		}

	} else {
		MPI_Recv(&start, 1, MPI_INT, 0, 0,
				MPI_COMM_WORLD, &status);
		MPI_Recv(&end, 1, MPI_INT, 0, 0,
				MPI_COMM_WORLD, &status);

	}
	printf("Sono il rank %d e mi è stato dato start [%d] e end [%d] \n",
			my_rank, start, end);
	iter = 0;

	//prendo il tempo di inizio dell'esecuzione parallela
	t1 = MPI_Wtime();

	//Inizio il calcolo di Laplace
	while (iter < 100 && flag == 0){
		//Qui si effettua il trasferimento delle righe boundary fra processori
		if (my_rank < p-1){
			MPI_Send(&x[(end-1)*nc], nc, MPI_DOUBLE, my_rank + 1, 0,
					MPI_COMM_WORLD );
		}
		if (my_rank > 0){

			MPI_Recv(&x[(start-1)*nc], nc, MPI_DOUBLE, my_rank - 1, 0,
					MPI_COMM_WORLD, &status );
		}

		if (my_rank > 0)
			MPI_Send(&x[(start)*nc], nc, MPI_DOUBLE, my_rank - 1, 1,
					MPI_COMM_WORLD );
		if (my_rank < p - 1)
			MPI_Recv(&x[(end)*nc], nc, MPI_DOUBLE, my_rank + 1, 1,
					MPI_COMM_WORLD, &status );

		//Calcolo equazione di Laplace
		for (i = start; i < end; i++) {
			for (j = 1; j < col; j++) {
				xnew[i*nc+j] = (x[(i + 1)*nc+j] + x[(i - 1)*nc+j] + x[i*nc+(j + 1)]
				+ x[i*nc+(j - 1)]) / 4;
			}
		}


		//calcolo dell'errore e copia dei risultati nella matrice x
		diffnorm = 0.0;
		for (i = start; i < end; i++) {
			for (j = 1; j < col; j++) {
				diffnorm += (xnew[i*nc+j] - x[i*nc+j]) * (xnew[i*nc+j] - x[i*nc+j]);
				x[i*nc+j] = xnew[i*nc+j];
			}
		}

		//ottengo tramite chiamata collettiva il risultato sommativo di diffnorm
		MPI_Allreduce(&diffnorm, &dif, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		//Sfrutto la chiamata All reduce a differenza della reduce poiché mi serve che ogni processore conosca il contenuto della variabile "dif"
		dif = sqrt( dif );
		if (my_rank == 0) {
			printf("DIF [%f] all'iterazione [%d] \n", dif, iter);
		}
		if(dif < 1.0e-2){
			t2 = MPI_Wtime();
			flag = 1;
		}
		iter++;
	}

	//Stampo su console i risultati dell'iterazione
	if(my_rank==0){
		if(flag==0){
			t2 = MPI_Wtime();
			printf("Esecuzione terminata con numero di iterazioni maggiore di 100\n");
		}
		else printf("Esecuzione terminata con numero di iterazioni minore di 100\n");
		printf( "Tempo di esercuzione  [%f]\n", t2 - t1 );
	}
	MPI_Finalize();

	return 0;
}

