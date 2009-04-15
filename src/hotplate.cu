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

    oldPlate_h = (float*) calloc(PLATE_AREA, sizeof(float));
    newPlate_h = (float*) calloc(PLATE_AREA, sizeof(float));

    cudaMalloc((void**) &oldPlate_d, PLATE_AREA);
    cudaMalloc((void**) &newPlate_d, PLATE_AREA);

    return 0;
}

