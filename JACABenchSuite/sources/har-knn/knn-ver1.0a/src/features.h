/**
* 
* Faculty of Engineering of the University of Porto (FEUP)
* Porto, Portugal
* December 2024
*/

#ifndef HAR_FEATURES_H 
#define HAR_FEATURES_H

#define MAX_ACCEL_VAL 20
#define MIN_ACCEL_VAL (-20)

#ifndef INT_MIN
#define INT_MIN -2147483648
#endif

#ifndef INT_MAX
#define INT_MAX 2147483647
#endif

void minmax(DATA_TYPE *min, DATA_TYPE *max, int num_points, Point *known_points, 
			int num_features);
			
void minmax_normalize(DATA_TYPE *min, DATA_TYPE *max, int num_points, Point *points, 
					int num_features);
					
void minmax_normalize_point(DATA_TYPE *min, DATA_TYPE *max, Point *point, int num_features);

#endif //HAR_FEATURES_H