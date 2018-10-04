
# Progetto    : "A simple Jacobi iteration"

## Installazione 
Dall'elenco corrente aprire il terminale ed estrarre l'archivio tramite il comando:

	tar -xzvf EdilioMassaro.tar.gz

Compilare il programma digitando:

	mpicc EdilioMassaro.c -o EdilioMassaro -lm

Eseguire il programma digitando:

	mpirun -np **numeroprocessori** EdilioMassaro

==**numero processori** è un numero intero che rappresenta il numero di core mpi.==

La matrice di input di prova è di dimensione 5000x5000. La matrice presa in esame per i risultati di strong scaling è di 10.000x10.000. Tale input è stato permesso tramite l'ausilio delle macchine M4 Large di AWS.

## Problema e scelte progettuali

#### Introduzione al problema:

Il problema da svolgere è di computare la soluzione dell'equazione di Laplace con finite differenze. In questo caso però non sarà necessario considerare tutta la matrice, ma considereremo delle celle boundary (ovvero quelle indicizzate in x[0][j], x[n-1][j], x[i][0], x[i][n-1]) come celle read-only, ovvero i processori non andranno a computare le soluzioni in quelle locazioni. 
Semplificherà notevolmente il calcolo poiché non sarà necessario computare la matrice come toroide. Ovviamente questa risulta una semplificazione del problema, che ci permette di avvicinarci alla soluzione reale con un determinato errore.


#### Input del problema:
L'input che andiamo a considerare sarà una matrice allocata dinamicamente come un array 1d di n righe ed nc colonne. Più una seconda matrice che conterrà i valori calcolati ad ogni iterazione tramite il calcolo dell'equazione di Laplace.
Le matrici vengono popolate da ogni processore tramite la funzione random con un seme uguale, così che potranno lavorare su matrici contenenti gli stessi valori in input.
La scelta di considerarle come array 1d è dovuta al problema dell'allocazione non contigua della memoria data dai 2d.



#### Divisione del carico:
Il processo di rank 0 si occuperà di dividere le righe della matrice **X** fra i **P** processori. Il valore rowNumber rappresenta il carico omogeneo di lavoro fra i p processori. Inoltre, nel caso che la divisione dia un resto diverso da 0, e che quindi vi sono delle righe rimanenti (remainder), allora un processo che ha un rank compreso fra 0 e work % p (ovvero fra 0 ed il numero del resto) gli verrà assegnata una riga in più su cui lavorare. Questo permetterà di ricoprire tutte le righe della matrice.
Quindi il master si calcolerà inizialmente i suoi indici di lavoro (start ed end) e dopo in base ad essi andrà man mano ad assegnare indici di inizio e fine a ciascun processore presente nel COMM_WORLD. Useremo la comunicazione point-to-point per segnalare gli indici di lavoro a ciascun processore.

```
for (int i = 1; i < p; i++) {
			tmpStart = tmpEnd;
			if (i < work % p) {
				tmpEnd = tmpStart + rowNumber+1;
			}
			else{
				tmpEnd = tmpStart + rowNumber;
			}
			MPI_Send(&tmpStart, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&tmpEnd, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
		}
```



##### Calcolo parallelo dell'iterazione di Jacobi
Ogni processore nella sua sotto matrice ha bisogno di conoscere i boundary immediatamente successivi e precedenti. Ovvero l'ultima riga del carico di lavoro del processo precedente e la prima riga del carico di lavoro del processo successivo. Visto che abbiamo considerato un'approssimazione del problema di Laplace significa che possiamo considerare la matrice non come un toroide. Per cui il processo 0 non ha bisogno di ricevere la riga superiore ed il processo di rank p-1 non ha bisogno di ricevere la riga inferiore (poiché sono le locazione di boundary read-only e mai modificate).


```
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
```


I primi if sono necessari appunto per il trasferimento delle righe di boundary (non di confine appunto). La scelta implementativa siffatta è stata pensata per evitare il trasferimento o la gather dell'intera matrice ad ogni iterazione per non saturare la banda. 

```
for (i = start; i < end; i++) {
			for (j = 1; j < col; j++) {
				xnew[i*nc+j] = (x[(i + 1)*nc+j] + x[(i - 1)*nc+j] + x[i*nc+(j + 1)]
						+ x[i*nc+(j - 1)]) / 4;
			}
		}
```

##### Reduce del risultato
La **MPI_Allreduce** è stata sfruttata poichè è necessario che non solo il processo 0 conosca il risultato della somma delle differenze (per il calcolo della convergenza), bensì ogni processore deve essere notificato del valore cumulato di diff per valutare se uscire dall'iterazione nel caso l'errore sia diventato minore di 1.0e-2.



##### MPI_Wtime:
La funzione è stata usata per calcolare il tempo di esecuzione della parte interessata, ovvero del calcolo di Laplace. Unicamente il processo di rank 0 si occuperà di stampare il suo tempo di fine.

## SCALING PERFORMANCE
##### STRONG SCALING
Per ogni tipologia di test verranno eseguiti più tentativi ed infine calcolato il tempo medio di esecuzione. I tempi vengono approssimati fino alla terza cifra decimale. I test sono stati effettuati tenendo conto in input una matrice di 10.000x10.000 generata con numeri casuali da 0 a 9. Il fattore di scalabilità è ricavato dalla formula ==t1/(N ** tn)*100%== :
1. t1: tempo di esecuzione a singolo processore.
1. N: numero di processori mpi usati
1. tn: tempo di esecuzione usando N processori.

Il test di strong scaling permette di trovare un punto che consenta di completare il calcolo in tempo ragionevole, ma non sprechi troppi cicli a causa del sovraccarico parallelo dei processori. 

I risultati dello strong scaling sono riportati nella seguente tabella.

| Core MPI | Tempo (ms) |  Percentuale   |
|--------|--------|--------|
|1        |    138346   |	//	|
|2        |    69407    |	99,6%  |
|4        |    35122    |	98,5%	   |
|6        |    23595    |	97,7%	   |
|8        |    17884    |	96,7%	   |
|10        |    24917   |	55,5%	   |
|12       |    21001    |	54,25%	   |
|14        |    18228   |	54,21%	   |
|16        |    16235   |	53,6%	   |

Riporto il grafico dei risultati di seguito.

![](https://raw.githubusercontent.com/Edilio1995/ASimpleJacobiPcPc/master/STRONG.PNG)

E' possibile notare un lieve aumento dei tempi quando si utilizzano 10 processori, si tratta del punto in cui si ha overhead di comunicazione che va ad impattare sulle performance. Lo stesso calo è notabile anche nei risultati del weak scaling nell'esatto punto.

##### WEAK SCALING
Per effettuare il weak scaling è stato fissato un valore di righe su cui ogni processore dovrà lavorare, ovvero di 5000 righe.Il fattore di scalabilità è ricavato dalla formula ==(t1 / tn)*100%== :
1. t1: tempo di esecuzione a singolo processore.
1. tn: tempo di esecuzione usando N processori.

I risultati dello weak scaling sono riportati nella seguente tabella.

| Core MPI | Tempo (ms) |  Percentuale   | 
|--------|--------|--------|
|1        |    34635   |	//	|
|2        |    35248    |	98,26%  |
|4        |    35831    |	99,43%	   |
|6        |    35117    |	98,62%	   |
|8        |    35138    |	98.57%	   |
|10        |    61190   |	56.68%	   |
|12       |    61481    |	56.33%	   |
|14        |    62267   |	55,65%	   |
|16        |    61836   |	56,01%	   |

![](https://raw.githubusercontent.com/Edilio1995/ASimpleJacobiPcPc/master/weak.PNG)

Esattamente nello stesso punto anche nel test della weak scaling è possibile notare il calo sempre dovuto all'overhead di comunicazione. 

#### Note:
1. Nella cartella pic sono contenute le due immagini dei grafici nel caso vi fossero problemi di visualizzazione.





