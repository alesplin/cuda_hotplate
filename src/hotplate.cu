/*
 * CUDA hotplate implementation
 */

#include "hotplate.h"
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

int main(int argc, char *argv[]) {
    /* stuff we'll need */
    float *oldPlate_d;
    float *newPlate_d;
    float *oldPlate_h;
    float *newPlate_h;
    float *tmpPlate_h;
    abool_t allSteady = FALSE;
    int iteration = 0;

    oldPlate_h = (float*) calloc(PLATE_AREA, sizeof(float));
    newPlate_h = (float*) calloc(PLATE_AREA, sizeof(float));

    cudaMalloc((void**) &oldPlate_d, PLATE_AREA);
    cudaMalloc((void**) &newPlate_d, PLATE_AREA);

    /* initialize plates */
    int x, y;
    for(y = 1; y < PLATE_SIZE - 1; y++) {
        for(x = 1; x < PLATE_SIZE - 1; x++) {
            oldPlate_h[LOC_H(x,y)] = WARM_START;
            newPlate_h[LOC_H(x,y)] = WARM_START;
        }
    }

    /* initialize the edges */
    for(x = 0; x < PLATE_SIZE; x++) {
        /* do the bottom edge */
        oldPlate_h[LOC_H(x,0)] = HOT_START;
        newPlate_h[LOC_H(x,0)] = HOT_START;
        /*printf("Column %d in row 0\n", LOC(x,0));*/
        /* do the left edge */
        oldPlate_h[LOC_H(0,x)] = COLD_START;
        newPlate_h[LOC_H(0,x)] = COLD_START;
        /*printf("Row %d in column 0\n", LOC(x,0));*/
        /* do the right edge */
        oldPlate_h[LOC_H(PLATE_SIZE-1,x)] = COLD_START;
        newPlate_h[LOC_H(PLATE_SIZE-1,x)] = COLD_START;
        /*printf("Row %d in column %d\n", LOC(x,0),PLATE_SIZE-1);*/
    }

    /* initialize our hot row */
    for(x = 0; x < FIXED_ROW_COL; x++) {
        oldPlate_h[LOC_H(x,FIXED_ROW)] = HOT_START;
        newPlate_h[LOC_H(x,FIXED_ROW)] = HOT_START;
    }

    /* initialize our lonely hot dot */
    oldPlate_h[LOC_H(DOT_X,DOT_Y)] = HOT_START;
    newPlate_h[LOC_H(DOT_X,DOT_Y)] = HOT_START;

    cudaMemcpy((void*)oldPlate_d, (void*) oldPlate_h, PLATE_AREA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)newPlate_d, (void*) newPlate_h, PLATE_AREA * sizeof(float), cudaMemcpyHostToDevice);

    while((!allSteady) && (iteration < MAX_ITERATION)) {
    /* run calculation kernel */
    
    /* synchronize */

    /* synchronize and run check kernel every other iteration */
    if(iteration ^ 1) { /* XOR faster than mod... */
        cudaThreadSynchronize();
    }

    /* swap plate pointers on the device... */

    /* increment iteration count */
    }

    return 0;
}

