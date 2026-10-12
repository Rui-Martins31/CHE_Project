#ifndef KNN_PARAMS_H //KNN_PARAMS_H
#define KNN_PARAMS_H

#include <float.h>
#include <stdint.h>

#ifndef SPECIALIZED
#define SPECIALIZED 1 // use functions according to the value of K
#endif

#ifndef USE_SQRT
#define USE_SQRT 1 // use sqrt in Euclidean distance
#endif

#ifndef MATH_TYPE
#define MATH_TYPE 1 // use the math.h functions according to the type float or double, e.g., sqrt or sqrtf 
#endif

#ifndef DIST_METHOD
#define DIST_METHOD 1 // calculate 1: Euclidean distance; 2: Manhattan distance
#endif 

#ifndef SCENARIO_FEATURES
#define SCENARIO_FEATURES 2
// 1 = WISDM_Act_v1.1 features; 2 = PAMAP2 features; 3 = gen100x8x5000; 4 = gen100x8x5000
#endif

#if SCENARIO_FEATURES == 1

#ifndef NUM_FEATURES
	#define NUM_FEATURES 43
#endif
#define NUM_CLASSES 6
#define NUM_KNOWN_POINTS 7383
#define NUM_NEW_POINTS 3163

#elif SCENARIO_FEATURES == 2

#ifndef NUM_FEATURES
	#define NUM_FEATURES 90
#endif
#define NUM_CLASSES 22
#define NUM_KNOWN_POINTS 6186
#define NUM_NEW_POINTS 1008

#elif SCENARIO_FEATURES == 3

#ifndef NUM_FEATURES
	#define NUM_FEATURES 100
#endif
#define NUM_CLASSES 8
#define NUM_KNOWN_POINTS 40002
#define NUM_NEW_POINTS 9998

#elif SCENARIO_FEATURES == 4

#ifndef NUM_FEATURES
	#define NUM_FEATURES 100
#endif
#define NUM_CLASSES 8
#define NUM_KNOWN_POINTS 8004
#define NUM_NEW_POINTS 1996

#endif

#ifndef K
#define K 3 // 3, 5 or 20 (some authors consider K=sqrt(NUM_TRAINING_INSTANCES)
#endif

#ifndef DT
#define DT 1 // 1: double; 2: float; 3: not used for now
#endif

#if DT == 1	//double
	#define DATA_TYPE double
	#define MAX_FP_VAL DBL_MAX
	#define MIN_FP_VAL -DBL_MAX
#elif DT == 2 //float
	#define DATA_TYPE float
	#define MAX_FP_VAL FLT_MAX
	#define MIN_FP_VAL -FLT_MAX
#endif

#if NUM_CLASSES > 128
	#define CLASS_ID_TYPE int16_t  // consider 0..32767 classes and -1 for unknown
#else
	#define CLASS_ID_TYPE int8_t  // consider 0..127 classes and -1 for unknown
#endif

typedef uint8_t K_TYPE;

#endif //KNN_PARAMS_H