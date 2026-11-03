/**
*
*  JACA Benchmark Suite v1.0
*
*	Human Activity Recognition (HAR) using a kNN for inference.
*
*	FEUP.DEI,
*	v1.0 January 2025
*	Author: João MP Cardoso
*	Email: jmpc@fe.up.pt
*
*	Scenario 1, WISDM dataset
*	Files for scenario must be copied from scenario-wisdm/
*	READ 2 // default 4
*	NUM_TRAINING_SAMPLES 7383
*	NUM_TESTING_SAMPLES 3163
*	NUM_FEATURES 43 
*	NUM_CLASSES 6
*
*	Scenario 2, PAMAP2 dataset
*	Files for scenario must be copied from scenario-pamap2/
*	READ 2 // default 4
*	NUM_TRAINING_SAMPLES 6186
*	NUM_TESTING_SAMPLES 1008
*	NUM_FEATURES 90 
*	NUM_CLASSES 22 // 21 activities + 1 activity representing transient activities
*
*	TIMING 0 or *1* // 0: w/o timing measurements; 1: w/ timing measurements; default: 1
*	ACCURACY 0 or *1* // 0: w/o calculating and reporting; 1: w/ calculation and reporting; default: 1
*	K = *3*, 5, 7 or 20 // any value >= 1 but these 4 were the ones considered for the scenarios; default: 3
* 	DATA_TYPE = float or double // default: double
*	DIMEM 0 or *1* // 0: w/o dynamic memory allocation; 1: w/ dynamic memory allocation; default: 0
*	STREAMING 0 or *1* // 0: w/o considering a possible streaming scheme; 1: considering streaming; default: 1
*
*	Other code versions:
*	SPECIALIZED 0 or 1 // 0: w/o specialization based on K value; 1: use specialized functions according to, for now, K=3; default: 1
*	MATH_TYPE 0 or 1 // // 0: use the math.h functions in the code; 1: use the math.h functions according to the type float or double, e.g., sqrt or sqrtf; default: 1
*	DIST_METHOD 0 or 1 // 1: Euclidean distance; 2: Manhattan distance; default: 1
*	USE_SQRT 0 or 1 // 0: w/o sqrt in Enclidean distance; 1: w/ sqrt in Euclidean distance; default: 1
*
*	All use minmax normalization of features
*
**/

/*
* Inputs
*	READ = 1			READ = 2			READ = 3			READ = 4
*	Static				static				From file			From file
*	-----------------------------------------------------------------------------
*	train1_notnorm.dat 	train1_norm.dat 	train2_notnorm.dat 	train2_norm.dat
*	-----------------------------------------------------------------------------
*	test1_notnorm.dat	test1_notnorm.dat	test2_notnorm.dat	test2_notnorm.dat
*/

#ifndef READ
	#define READ 4 // 1 and 3: static initalization; 2 and 4: read from .dat
#endif

#if READ == 3
#ifndef TRAIN_PATH
#define TRAIN_PATH "train2_notnorm.dat"
#endif
#ifndef TEST_PATH
#define TEST_PATH "test2_notnorm.dat"
#endif
#elif READ == 4
#ifndef TRAIN_PATH
#define TRAIN_PATH "train2_norm.dat"
#endif
#ifndef TEST_PATH
#define TEST_PATH "test2_notnorm.dat"
#endif
#endif

#ifndef TIMING
	#define TIMING 1 // 0: without; 1: Linux/Windows, 2: specific timers (not implemented)
#endif

#ifndef VERIFY
	#define VERIFY 0 	// 0: none verification;
						// 1: to verify if the results are according to the ones expected
#endif

#ifndef ACCURACY
	#define ACCURACY 1 	// 0: no; 1: to report the accuracy of the classification
#endif

#ifndef STREAMING
	#define STREAMING 1
	// 0: all the points are first normalized and only then the classification occurs;
	// 1: each point is normalized in the loop of the classification
#endif

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "params.h"
#include "types.h"
#include "utils.h"
#include "io.h"
#include "knn.h"
#include "features.h"

#if TIMING == 1
	#include "timer.h"
#endif


#if READ == 1 // data embedded in program, arrays statically filled

	Point known_points1[NUM_KNOWN_POINTS] = {
					#include "train1_notnorm.dat"
				}; //{{{1,2,3,4,5},1}};
	Point *known_points = known_points1;

	Point new_points1[NUM_NEW_POINTS] = {
					#include "test1_notnorm.dat"
				}; //{{{2.0,1.0, 1.0},0},{{2.0,3.0,2.0},0},...};
	Point *new_points = new_points1;

#elif READ == 2 // data embedded in program, arrays statically filled

	Point known_points1[NUM_KNOWN_POINTS] = {
					#include "train1_norm.dat"
				}; //{{{1,2,3,4,5},1}};
	Point *known_points = known_points1;

	Point new_points1[NUM_NEW_POINTS] = {
					#include "test1_notnorm.dat"
				}; //{{{2.0,1.0, 1.0},0},{{2.0,3.0,2.0},0},...};
	Point *new_points = new_points1;

#elif READ == 3 || READ == 4 // data read from files

	#if DIMEM == 0
	Point known_points1[NUM_KNOWN_POINTS];
	Point *known_points = known_points1;

	Point new_points1[NUM_NEW_POINTS];
	Point *new_points = new_points1;

    #else
	Point *known_points
	Point *new_points;

	#endif
#endif


#if VERIFY == 1  // verify if classifications are still as the golden ones
	CLASS_ID_TYPE key[NUM_NEW_POINTS] = {
		#if K == 20
			#include "key-READ4-k20.dat"
		#elif K == 3
			#include "key-READ4-k3.dat"
		#elif K == 5
			#include "key-READ4-k5.dat"
		#elif K == 7
			#include "key-READ4-k7.dat"
		#endif
	};
#endif

int main() {

	if (getenv("OMP_NUM_THREADS") == NULL)
		omp_set_num_threads(16);

	const K_TYPE k = K;
	const int num_features = NUM_FEATURES;
    const int num_classes = NUM_CLASSES;
    const int num_known_points = NUM_KNOWN_POINTS;
	const int num_new_points = NUM_NEW_POINTS;
	
	#if DIMEM == 1 && (READ == 3 || READ == 4)
	known_points = (Point *) malloc(NUM_KNOWN_POINTS*sizeof(Point));
	new_points = (Point *) malloc(NUM_NEW_POINTS*sizeof(Point));
	#endif
	
	#if READ == 1 || READ == 2 // data embedded in program
		printf("All statically initialized from data in files\n");
	#elif READ == 3  // data read from .dat files
		printf("Initializing non-normalized training vectors of features from .dat ...\n");
		read_data_points(TRAIN_PATH, num_features, num_known_points, known_points);
		printf("Initializing testing vectors of features from .dat ...\n");
		read_data_points(TEST_PATH, num_features, num_new_points, new_points);
		printf("Initialization done.\n\n");
	#elif READ == 4  // data read from .dat files
		printf("Initializing normalized training vectors of features from .dat ...\n");
		read_data_points(TRAIN_PATH, num_features, num_known_points, known_points);
		printf("Initializing testing vectors of features from .dat ...\n");
		read_data_points(TEST_PATH, num_features, num_new_points, new_points);
		printf("Initialization done.\n\n");
	#endif
	
	#if READ == 2 || READ == 4 // load min and max feature values for normalization
		DATA_TYPE min[NUM_FEATURES] = {
				#include "min.dat"
			};
		DATA_TYPE max[NUM_FEATURES] = {
				#include "max.dat"
			};
	#elif READ == 1 || READ == 3 // calculate min and max feature values for normalization
		DATA_TYPE min[NUM_FEATURES];
		DATA_TYPE max[NUM_FEATURES];
		
		// determine min and max from known points
		minmax(min, max, num_known_points, known_points, num_features);

		// normalize known points
		minmax_normalize(min, max, num_known_points, known_points, num_features);
		
	#endif
	
	#if STREAMING == 0 // no streaming: normalize all features of test vectors before 
		minmax_normalize(min, max, num_new_points, new_points, num_features);
	#endif
	
    printf("Executing kNN...\n");

	#if TIMING == 1
	Timer *timer = timer_init();
	timer_start(timer);
	#endif
	
	#if ACCURACY == 1
	int fail = 0; // count the number of test instances incorrectly classified
	#endif
		
	// loop over the input instances to classify.
	// Note that depending on the application this can be
	// instances arriving as streaming data.
	// Here assume that the loop below needs to run in serial mode and the
	// value of num_new_points is just to test
    // Classify KNN_BATCH instances per parallel pass over the training set.
    // Classes are still emitted in arrival order. This waits for a batch
    // before classifying, so it is not one-instance-at-a-time streaming.
    for (int i = 0; i < num_new_points; ) {
		int nbatch = num_new_points - i;
		if (nbatch > KNN_BATCH)
			nbatch = KNN_BATCH;

		for (int t = 0; t < nbatch; t++) {
			Point *new_point = &new_points[i + t];
			#if STREAMING == 1
			minmax_normalize_point(min, max, new_point, num_features);
			#endif
		}

		CLASS_ID_TYPE classes[KNN_BATCH];
		#if SPECIALIZED == 1 && K == 3
		knn_classify_batch_3(&new_points[i], nbatch,
			known_points, num_known_points, num_features, classes);
		#else
		knn_classify_batch(&new_points[i], nbatch, k, num_classes,
			known_points, num_known_points, num_features, classes);
		#endif

		for (int t = 0; t < nbatch; t++) {
			Point *new_point = &new_points[i + t];
			CLASS_ID_TYPE instance_class = classes[t];

			#if ACCURACY
				if(new_point->classification_id != instance_class) fail++;
			#endif

			put(&instance_class, 0, "class id:"); // 0 is the id used for char

			#if STREAMING == 0
			new_point->classification_id = instance_class;
			#endif
		}
		i += nbatch;
    }

	#if TIMING == 1
		timer_stop(timer);
	#endif

    printf("kNN done.\n\n");

    printf("HAR: number of features = %d\n", num_features);
    printf("HAR: number of classes = %d\n", num_classes);
	
	printf("kNN: k = %d\n", k);
    printf("kNN: number of training instances = %d\n", num_known_points);
    printf("kNN: number of testing instances = %d\n", num_new_points);

	#if DT == 2
      printf("kNN: data type used = float\n");
    #elif DT == 1
      printf("kNN: data type used = double\n");
	#endif
	
    printf("kNN: number of classified instances = %d\n", num_new_points);

	#if ACCURACY == 1
		printf("kNN: number of classifications wrong = %d\n", fail);
		printf("kNN: number of classifications right = %d\n", num_new_points-fail);
		printf("kNN: accuracy = %.2f %s\n\n", 100*((float)(num_new_points-fail)/(float) num_new_points),"%");
	#endif

	#if VERIFY == 1
		verify_results(num_new_points, new_known_points, key);
	#endif

	#if TIMING == 1
		const double time = timer_get_s(timer);
		timer = timer_destroy(timer);
		
		printf("\nTotal execution time:  %.4f s\n", time);
		
		printf("Inference time: %.4f ms\n", (time/num_new_points*1000));
		printf("Throughput: %.2f inferences/s\n", num_new_points/time);
	#endif

	#if DIMEM != 0
		free(new_points);
		free(known_points);
	#endif

    return 0;
}