/*
 * header for CUDA hotplate
 */

/*
 * some macros to help us
 */
#define PLATE_SIZE 1024
#define PLATE_AREA ( PLATE_SIZE * PLATE_SIZE )
#define FLOAT_ACCURACY 0.0001f /* for comparing floating point numbers */
#define STEADY_THRESHOLD 0.1f
#define HOT_START 100.0f
#define COLD_START 0.0f
#define WARM_START 50.0f
#define TRUE 1
#define FALSE 0
#define PRINT_LINE ( printf("(%d,%d) at %d\n", x,y, __LINE__) )

#define MAX_ITERATION 400

#define BLOCKS_Y 2
#define BLOCKS_X 2
#define THREADS_Y 512
#define THREADS_X 512

/* macro for checking fixed locations */
#define IS_FIXED(x,y) ( ((x == DOT_X) && (y == DOT_Y))?TRUE:((y == FIXED_ROW) && (x < FIXED_ROW_COL))?TRUE:FALSE )

/* magic numbers for the fixed hot dot and row */
#define DOT_X 199 /* row 200 */
#define DOT_Y 499 /* column 500 */
#define FIXED_ROW 399 /* row 400 */
#define FIXED_ROW_COL 329 /* last column of the fixed row */

/* macros for plate positioning/indexing */
#define LOC_H(x,y) ( (y*PLATE_SIZE)+x )
#define LEFT_LOC_H(x,y) ( (y*PLATE_SIZE)+(x-1) )
#define RIGHT_LOC_H(x,y) ( (y*PLATE_SIZE)+(x+1) )
#define LOWER_LOC_H(x,y) ( ((y-1)*(PLATE_SIZE))+x )
#define UPPER_LOC_H(x,y) ( ((y+1)*(PLATE_SIZE))+x )

/* take care of device emulation... */
#ifdef __DEVICE_EMULATION__
#define EMUSYNC __syncthreads()
#else
#define EMUSYNC
#endif

/* a boolean type */
typedef int abool_t;

/* timing function */
double getTime();

