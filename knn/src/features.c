/**
* F
*
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "types.h"
#include "params.h"
#include "utils.h"
#include "features.h"

/*
* Determine the min and max values for each feature for a set of 
* points.
*/ 
//__attribute__ ((optimize("-O0"))) 
void minmax(DATA_TYPE *min, DATA_TYPE *max, int num_points, Point *known_points, int num_features) {

	for (int j = 0; j < num_features; j++) {
		min[j] = MAX_FP_VAL;
		max[j] = MIN_FP_VAL;
		//printf("%e, %e\n", MIN_FP_VAL, MAX_FP_VAL);
	}
	
	for (int i = 0; i < num_points; i++) {
		for (int j = 0; j < num_features; j++) {
            if(known_points[i].features[j] < min[j]) 
				min[j] = known_points[i].features[j];
            if(known_points[i].features[j] > max[j]) 
				max[j] = known_points[i].features[j];
        }
    }
	
}

/*
* Normalize the features of each point using minmax normalization.
*/
void minmax_normalize(DATA_TYPE *min, DATA_TYPE *max, int num_points, Point *points, int num_features)
{

    for (int i = 0; i < num_points; i++) {
        for (int j = 0; j < num_features; j++) {
			
			DATA_TYPE nfeature = (DATA_TYPE) ((points[i].features[j] - min[j])/(max[j] - min[j]));
			
			// in case the normalization returns a NaN or INF
			if(isnan(nfeature)) nfeature = (DATA_TYPE) 0.0;
			else if(isinf(nfeature)) nfeature = (DATA_TYPE) 1.0;
			
			points[i].features[j] = nfeature;
		}
		//show_point(points[i], num_features); 
    }
}

/*
* Normalize the features of a single point using minmax normalization.
*/
void minmax_normalize_point(DATA_TYPE *min, DATA_TYPE *max, Point *point, int num_features) {

    for (int j = 0; j < num_features; j++) {
			
		DATA_TYPE nfeature = (DATA_TYPE) ((point->features[j] - min[j])/(max[j] - min[j]));
			
		// in case the normalization returns a NaN or INF
		if(isnan(nfeature)) nfeature = (DATA_TYPE) 0.0;
		else if(isinf(nfeature)) nfeature = (DATA_TYPE) 1.0;
			
		point->features[j] = nfeature;
    }
}
