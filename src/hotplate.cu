/*
 * CUDA hotplate implementation
 */

#include "hotplate.h"
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

/* calculation kernel */
__global__ void runCalc(float *old_d, float *new_d) {
    /* get our thread's coordinates */
    int y = (blockIdx.y*blockDim.y) + threadIdx.y;
    int x = (blockIdx.x*blockDim.x) + threadIdx.x;
    /*printf("thread (%d,%d) calculating in...\n", x,y);*/

    /* bail if we're on an edge... */
    if((x == 0) || (x == PLATE_SIZE - 1) || (y == 0) || (y == PLATE_SIZE - 1)) {
        return;
    }

    /*PRINT_LINE;*/
    /* calculate my spot and bail */
    if(!IS_FIXED(x,y)) {
        new_d[LOC_H(x,y)] = (float)(old_d[LEFT_LOC_H(x,y)]
                        + old_d[RIGHT_LOC_H(x,y)]
                        + old_d[LOWER_LOC_H(x,y)]
                        + old_d[UPPER_LOC_H(x,y)]
                        + 4 * old_d[LOC_H(x,y)] ) / 8;
    }
}

/* check to see if we're in a "steady" state */
__global__ void runCheck(float *old_d, float *new_d, abool_t *allSteady_d) {
    /* get our coordinates */
    int y = (blockIdx.y*blockDim.y) + threadIdx.y;
    int x = (blockIdx.x*blockDim.x) + threadIdx.x;
    /*printf("thread (%d,%d) checking...\n", x,y);*/

    /* bail if we're on an edge... */
    if((x == 0) || (x == PLATE_SIZE - 1) || (y == 0) || (y == PLATE_SIZE - 1)) {
        return;
    }

    /* check my spot; if not steady, set the allSteady to false */
    if((*allSteady_d == TRUE) && (!IS_FIXED(x,y))) {
        float me = new_d[LOC_H(x,y)];
        float neighborAvg = (new_d[LEFT_LOC_H(x,y)]
                        + new_d[RIGHT_LOC_H(x,y)]
                        + new_d[LOWER_LOC_H(x,y)]
                        + new_d[UPPER_LOC_H(x,y)]) / 4;
        if(fabsf(me - neighborAvg) >= STEADY_THRESHOLD) {
            *allSteady_d = FALSE;
            /*printf("(%d,%d) set allSteady_d to: %d\n", x,y,*allSteady_d);*/
        }
    } /* END if not steady and not fixed */
}

/* check for steady kernel */

int main(int argc, char *argv[]) {
    /* stuff we'll need */
    double start;
    double end;
    double et;
    float *oldPlate_d;
    float *newPlate_d;
    float *oldPlate_h;
    float *newPlate_h;
    float *tmpPlate_d;
    abool_t *allSteady_h;
    abool_t *allSteady_d;
    int iteration = 0;

    oldPlate_h = (float*) calloc(PLATE_AREA, sizeof(float));
    newPlate_h = (float*) calloc(PLATE_AREA, sizeof(float));
    allSteady_h = (abool_t*) calloc(1,sizeof(abool_t));

    cudaMalloc((void**) &oldPlate_d, PLATE_AREA * sizeof(float));
    cudaMalloc((void**) &newPlate_d, PLATE_AREA * sizeof(float));
    cudaMalloc((void**) &allSteady_d, sizeof(abool_t));

    /* initialize plates */
        int x, y;
        printf("main at %d...\n", __LINE__);
        for(y = 1; y < PLATE_SIZE - 1; y++) {
            for(x = 1; x < PLATE_SIZE - 1; x++) {
                oldPlate_h[LOC_H(x,y)] = WARM_START;
                newPlate_h[LOC_H(x,y)] = WARM_START;
            }
        }

        printf("main at %d...\n", __LINE__);
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

        printf("main at %d...\n", __LINE__);
        /* initialize our hot row */
        for(x = 0; x < FIXED_ROW_COL; x++) {
            oldPlate_h[LOC_H(x,FIXED_ROW)] = HOT_START;
            newPlate_h[LOC_H(x,FIXED_ROW)] = HOT_START;
        }

        printf("main at %d...\n", __LINE__);
        /* initialize our lonely hot dot */
        oldPlate_h[LOC_H(DOT_X,DOT_Y)] = HOT_START;
        newPlate_h[LOC_H(DOT_X,DOT_Y)] = HOT_START;

    printf("main at %d...\n", __LINE__);
    cudaMemcpy((void*)oldPlate_d,
                (void*) oldPlate_h,
                PLATE_AREA * sizeof(float),
                cudaMemcpyHostToDevice);


    printf("main at %d...\n", __LINE__);
    cudaMemcpy((void*)newPlate_d,
                (void*) newPlate_h,
                PLATE_AREA * sizeof(float),
                cudaMemcpyHostToDevice);
    

    /* get our grids/blocks all ready... */
        dim3 calcGrid;
        calcGrid.x = BLOCKS_X;
        calcGrid.y = BLOCKS_Y;
        dim3 calcBlock;
        calcBlock.x = THREADS_X;
        calcBlock.y = THREADS_Y;

        dim3 checkGrid;
        checkGrid.x = BLOCKS_X;
        checkGrid.y = BLOCKS_Y;
        dim3 checkBlock;
        checkBlock.x = THREADS_X;
        checkBlock.y = THREADS_Y;

    start = getTime();
    while((*allSteady_h != TRUE) && (iteration < MAX_ITERATION)) {
        /*printf("main at %d...\n", __LINE__);*/


        /* run calculation kernel */
        /*printf("main at %d...\n", __LINE__);*/
        runCalc<<<calcGrid,calcBlock>>>(oldPlate_d, newPlate_d);
        /* synchronize */

        /* synchronize and run check kernel every other iteration */
        if(iteration ^ 1) { /* XOR faster than mod... */
            *allSteady_h = TRUE;
            printf("main set allSteady_h to %d\n", *allSteady_h);
            cudaMemcpy(allSteady_d,
                    allSteady_h,
                    sizeof(abool_t),
                    cudaMemcpyHostToDevice);
            cudaThreadSynchronize();
            runCheck<<<checkGrid,checkBlock>>>(oldPlate_d, newPlate_d, allSteady_d);
        }

        cudaThreadSynchronize();
        /* swap plate pointers on the device... */
        tmpPlate_d = oldPlate_d;
        oldPlate_d = newPlate_d;
        newPlate_d = tmpPlate_d;
        
        /* increment iteration count */
        iteration++;
        cudaMemcpy(allSteady_h,
                allSteady_d,
                sizeof(abool_t),
                cudaMemcpyDeviceToHost);
    }
    end = getTime();
    et = end - start;
    printf("%d iterations in %0.4f seconds...\n", iteration, et);

    free(oldPlate_h);
    free(newPlate_h);
    cudaFree(oldPlate_d);
    cudaFree(newPlate_d);

    return 0;
}

/*
 * get a high-precision representation of the current time
 */
double getTime() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1e-6;
}

