/*
 * Task 4: applies convolution with an edge-detection filter to a bmp
 *  image in parallel. Uses MPI.
 *
 * Author: Nathaniel Cantwell
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"
#define ISIZE 256
#define BSIZE 32
#define FSIZE 3
#define BROWS 1
#define BCOLS 1
#define ROOT 0

#define RDEBUG 22
#define CDEBUG 50

void process(FILE* in, FILE* out, int size, unsigned char header[], unsigned char pixel[][ISIZE+2]);
int timedif(struct timeval start, struct timeval end);
void quit(void);
void serialProcess(unsigned char img[][ISIZE+2], int filter[][FSIZE], unsigned char result[][ISIZE]);
void syncPrint(int taskid, int numTasks, unsigned char arr[][ISIZE/BCOLS+2]);

int main(int argc, char* argv[]) {
	// Image matrix. +2 for zero padding.
	unsigned char pixel[ISIZE+2][ISIZE+2];

	// Image block matrix and result block matrix. +2 for padding with zeros and values
	unsigned char myBlock[BSIZE+2][BSIZE+2] = {0};

	// Full result image block matrix
	unsigned char result[ISIZE][ISIZE];
	unsigned char chkResult[ISIZE][ISIZE];

	// BMP image header
	unsigned char header[2000];

	// Input and output filenames
	char filein[200] = "/gpfs/home/n/a/nadacant/BigRed2/Tasks/Task4/openmpi.bmp";
	char fileout[200] = "/gpfs/home/n/a/nadacant/BigRed2/Tasks/Task4/openmpiout.bmp";
	//char filein[200] = "/home/nate/Code/Cstuff/ece595/Task4/openmpi.bmp";
	//char fileout[200] = "/home/nate/Code/Cstuff/ece595/Task4/openmpiout.bmp";

	// Convolution filter
	int filter[FSIZE][FSIZE] = {
		{ -1, -1, -1 },
		{ -1, 8, -1 },
		{ -1, -1, -1 }
	};

	FILE *fin;
	FILE *fout;
	int offset, i, j, k, l, task, row, col, block;
	int taskid, numTasks;
	MPI_Status stat;
	long serialTime, parallelTime;

	// Datatypes for block matrix variants
	MPI_Datatype tempblock, blockType, tempPad, padBlockType;

	// Compute number of blocks based on ISIZE and BSIZE
	int numBlocks = ISIZE*ISIZE/(BSIZE*BSIZE);
	// Number of block columns. Should be 8 with 256x256 img and 32x32 partitons
	int bCols = ISIZE/BSIZE;

	// Timing structs
	struct timeval start, end;

	// Initialize MPI environment
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

	// Initialize pixel array
	for (i = 0; i < ISIZE+2; i++) {
		for (j = 0; j < ISIZE+2; j++) {
			pixel[i][j] = 0;
		}
	}

	// Create block matrix subarray type (not padded with 0's)
	// Dimensions of matrix
	int sizes[2] = {ISIZE, ISIZE};
	int padSizes[2] = {ISIZE+2, ISIZE+2};
	// Dimensions of blocks
	int blocksizes[2] = {BSIZE, BSIZE}; 	
	int padBlockSizes[2] = {BSIZE+2, BSIZE+2}; 	
	// Starting position. Change pointer rather than this.
	int starts[2] = {0,0};
	
	// Create type with correct size, subsize, lower bound, order, atomic type, and type container.
	MPI_Type_create_subarray(2, sizes, blocksizes, starts, MPI_ORDER_C, MPI_BYTE, &tempblock);
	// Reize type to change extent
	MPI_Type_create_resized(tempblock, 0, BSIZE*sizeof(unsigned char), &blockType);
	// Commit the type.
	MPI_Type_commit(&blockType);

	// Create padded block matrix with corrected sizing parameters

	// Create padded block matrix type with correct size, subsize, lower bound, order, atomic type, and type container.
	MPI_Type_create_subarray(2, padSizes, padBlockSizes, starts, MPI_ORDER_C, MPI_BYTE, &tempPad);
	// Reize type to change extent
	MPI_Type_create_resized(tempPad, 0, (BSIZE+2)*sizeof(unsigned char), &padBlockType);
	// Commit the padded block type
	MPI_Type_commit(&padBlockType);

	// Main master/slave branch
	if (taskid == ROOT) {
		// Sync for timing
		MPI_Barrier(MPI_COMM_WORLD);
		gettimeofday(&start, NULL);

		// Read input file
		fin = fopen(filein, "rb");
		if (fin == NULL) {
			printf("Error opening input file: %s\n", filein);
			quit();
		}

		// Open output file
		fout = fopen(fileout, "wb");
		if (fout == NULL) {
			printf("Error opening output file: %s\n", fileout);
			quit();
		}

		// Ignore beginning of header
		fread(&(header[0]), 1, 10, fin);

		// Offset to first pixel of image
		fread(&(offset), 4, 1, fin);
		printf("offset %d\n", offset);

		// Rewind file to beginning
		rewind(fin);
		// Read pixels into array
		process(fin, fout, offset, header, pixel);
		// Close input file
		fclose(fin);

		// Get length of serial execution
		gettimeofday(&end, NULL);
		serialTime = timedif(start, end);

		// Time parallel portion of code
		MPI_Barrier(MPI_COMM_WORLD);
		gettimeofday(&start, NULL);

		// Case where there are no slaves.
		if (numTasks == 1) {
			serialProcess(pixel, filter, result);
		} else {
			// Variable to define if sending is finished
			int quit = 0;

			// Initialize current block number
			block = 0;

			// Send block number and block data to all processes until no data remains
			while (!quit) {
				for (j = 1; j < numTasks; j++) {
					// Send block number to slave j
					MPI_Send(&block, 1, MPI_INT, j, 0, MPI_COMM_WORLD);

					// Send padded block to slave j
					row = (block / bCols) * BSIZE;
					col = (block % (bCols)) * BSIZE;
					MPI_Send(&pixel[row][col], 1, padBlockType, j, 0, MPI_COMM_WORLD);

					// Check if data remains
					if (++block >= numBlocks) {
						quit = 1;
						break;
					}
				}
			}

			// Send bad status to all blocks so they may exit
			block = -1;
			for (j = 1; j < numTasks; j++) {
				// Send -1 as block number to all processes
				MPI_Send(&block, 1, MPI_INT, j, 0, MPI_COMM_WORLD);
			}

			// Receive blocks from all processes
			for (block = 0; block < numBlocks; block++) {
				// Compute row/column of block to send.
				row = (block / bCols) * BSIZE;
				col = (block % (bCols)) * BSIZE;
				
				// Receive result blocks from all processes
				MPI_Recv(&result[row][col], 1, blockType, MPI_ANY_SOURCE, block, MPI_COMM_WORLD, &stat);
			}
		}

		// Sync for end of parallel timing
		MPI_Barrier(MPI_COMM_WORLD);
		gettimeofday(&end, NULL);
		parallelTime = timedif(start, end);
		
		// Time remaining portion of serial code
		gettimeofday(&start, NULL);

		// Output pixels to file. Start from 1 to account for zeros.
		for (i = 0; i < ISIZE; i++)
			for (j = 0; j < ISIZE; j++)
				fputc(result[i][j], fout);

		fclose(fout);

		// Time remaining portion of serial code
		gettimeofday(&end, NULL);
		serialTime += timedif(start, end);

		// Print timing information
		printf("Serial portion (ms): %lf\n", (double)serialTime/(double)1000);
		printf("Parallel portion (ms): %lf\n", (double)parallelTime/(double)1000);
		
	} else {
		// Timing synchronization (for reading input)
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);

		// List of blocks received/computed. Maximum is all blocks.
		int blockList[ISIZE*ISIZE/(BSIZE*BSIZE)];

		// Number of blocks for which the result has been computed
		int numComp;
		
		// Loop until master sends kill signal
		numComp = 0;
		while (1) {
			// Receive block number, encodes state
			MPI_Recv(blockList+numComp, 1, MPI_INT, ROOT, 0, MPI_COMM_WORLD, &stat);

			// If ROOT sends -1 as block, indicates no remaining data
			if (blockList[numComp] == -1) {
				break;
			}

			// Receive padded block from master process
			MPI_Recv(&myBlock[0][0], (BSIZE+2)*(BSIZE+2), MPI_BYTE, ROOT, 0, MPI_COMM_WORLD, &stat);

			// Compute appropriate top corner row/col of result
			row = (blockList[numComp] / bCols) * BSIZE;
			col = (blockList[numComp] % bCols) * BSIZE;
	
			// Compute convolution and store in result array
			for (i = 0; i < BSIZE; i++) {
				for (j = 0; j < BSIZE; j++) {
					result[row+i][col+j] = 0;
					for (k = 0; k < 3; k++) {
						for (l = 0; l < 3; l++) {
							//int temp = myBlock[i+k][j+l]*filter[k][l];
							/*
							if ((i+row) == RDEBUG && (j+col) == CDEBUG) {
								printf("result[%d][%d] == %d\n", row+i, col+j, result[row+i][col+j]);
								printf("temp = %d, cast temp = %d\n", temp, (unsigned char)temp);
							}
							*/
							result[row+i][col+j] = result[row+i][col+j] + myBlock[i+k][j+l]*filter[k][l];
							//if ((i+row) == RDEBUG && (j+col) == CDEBUG) printf("After additon, result[%d][%d] == %d\n\n", row+i, col+j, result[row+i][col+j]);
						}
					}
				}
			}

			// Update number of computed blocks
			numComp++;
		}

		// Send all result blocks to master, tagged with block number.
		for (k = 0; k < numComp; k++) {
			// Compute row/column of block to send.
			row = (blockList[k] / bCols) * BSIZE;
			col = (blockList[k] % bCols) * BSIZE;
			
			// Send result block to ROOT
			MPI_Send(&result[row][col], 1, blockType, ROOT, blockList[k], MPI_COMM_WORLD);
		}

		// Timing synchronization
		MPI_Barrier(MPI_COMM_WORLD);
	}

	// Exit MPI environment, free resources
	MPI_Type_free(&padBlockType);
	MPI_Type_free(&blockType);
	MPI_Finalize();
	return 0;
}

// read header from input file and write it to output file
// read input matrix
void process(FILE* in, FILE* out, int size, unsigned char header[], unsigned char pixel[][ISIZE+2]) {
	int i, j;
	for (i = 0; i < size; i++) { // Header
		header[i] = fgetc(in);
		fputc(header[i], out);
	}
	for (i = 1; i < ISIZE+1; i++)
		for (j = 1; j < ISIZE+1; j++)
			pixel[i][j] = fgetc(in);
}

// Calculates the time difference between two different struct timeval structures. Returns the number of microseconds.
int timedif(struct timeval start, struct timeval end) {
	return (end.tv_sec*1000000 + end.tv_usec) - (start.tv_sec*1000000 + start.tv_usec);
}

// Process the image in serial. Used for error-checking or if numTasks is 1
void serialProcess(unsigned char img[][ISIZE+2], int filter[][FSIZE], unsigned char result[][ISIZE]) {
	int i, j, k, l;

	// Compute convolution and store in result
	for (i = 0; i < ISIZE; i++) {
		for (j = 0; j < ISIZE; j++) {
			result[i][j] = 0;
			for (k = 0; k < 3; k++) {
				for (l = 0; l < 3; l++) {
					result[i][j] += img[i+k][j+l]*filter[k][l];
				}
			}
		}
	}
}

// Exits the program. Frees MPI resources.
void quit(void) {
	MPI_Finalize();
	exit(1);
}

