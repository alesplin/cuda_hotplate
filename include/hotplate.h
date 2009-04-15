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
#define PRINT_LINE ( printf("Thread %d, at line %d\n", iproc, __LINE__) )

/* macro for checking fixed locations */
#define IS_FIXED(x,y) ( ((x == DOT_X) && (y == DOT_Y))?TRUE:((y == FIXED_ROW) && (x < FIXED_ROW_COL))?TRUE:FALSE )

/* macros for plate positioning/indexing */
#define LOC(x,y) ( (y*plateSize)+x )
#define LEFT_LOC(x,y) ( (y*plateSize)+(x-1) )
#define RIGHT_LOC(x,y) ( (y*plateSize)+(x+1))
#define LOWER_LOC(x,y) ( ((y-1)*(plateSize))+x )
#define UPPER_LOC(x,y) ( ((y+1)*(plateSize))+x )

/* take care of device emulation... */
#ifdef __DEVICE_EMULATION__
#define EMUSYNC __syncthreads()
#else
#define EMUSYNC
#endif

