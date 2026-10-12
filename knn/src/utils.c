
#include <stdio.h>
#include <float.h>
#include <stdlib.h>

#include "utils.h"


/*
* Verify if the classifications equal the original ones stored in key
*/
void verify_results(int num_new_points, const Point *new_points, const CLASS_ID_TYPE *key) {
	
    if (key == NULL) {
        printf("Skipping verification.\n");
        return;
    }

    int passed = 1;
    printf("Verifying results...\n");
    for (int i = 0; i < num_new_points; ++i) {

        CLASS_ID_TYPE classified = new_points[i].classification_id;
        CLASS_ID_TYPE truth = key[i];

        if (classified == truth) {
            printf(" %d %s %d\n", classified, "=", truth);
        } else {
            printf(" %d %s %d\n", classified, "!=", truth);
            passed = 0;
        }

    }

    printf("Verification is complete: ");
    if (passed == 1) {
        printf("Passed!\n");
    } else {
        printf("Failed!\n");
    }
}


/*
* Show points
*/
void show_points(int num_points, Point *points, int num_features) {

    for (int i = 0; i < num_points; i++) {
		show_point(points[i], num_features);
    }
}

/*
* show the values of a point: features and class.
*/
void show_point(Point point, int num_features) {

    for (int j = 0; j < num_features; j++) {
		if(j == 0)
			printf("%.3f", point.features[j]);
		else
			printf(",%.3f", point.features[j]);
    }
    printf(", class = %d\n", point.classification_id);
}


/*
* Output points: 
* 	format = 1 => values seprated by a bank space
*	format ! 1 => output according to the array initialization of C
*/
void output_points(int num_points, Point *points, int num_features, int format) {

	if(format == 1) {
    for (int i = 0; i < num_points; i++) {
		output_point(points[i], num_features, format);
    }
	} else {
	for (int i = 0; i < num_points; i++) {
		printf("{{");
		output_point(points[i], num_features, format);
		printf("}");
		if(i != num_points-1) printf(",\n");
    }
	}
		
}

/*
* Output the values of a point: features and class.
*/
void output_point(Point point, int num_features, int format) {
	if(format == 1) {
		for (int j = 0; j < num_features; j++) {
			if(j == 0)
				printf("%.3f", point.features[j]);
			else
				printf(" %.3f", point.features[j]);
		}
		printf(" %d\n", point.classification_id);
	} else {
		for (int j = 0; j < num_features; j++) {
			if(j == 0)
				printf("%.3f", point.features[j]);
			else
				printf(",%.3f", point.features[j]);
		}
		printf("},%d", point.classification_id);
	}
}

void output_minmax(DATA_TYPE *min, DATA_TYPE *max, int num_features) {
	printf("{");
	for (int j = 0; j < num_features; j++) {
		if(j<num_features-1) printf("%.4f,",min[j]);
		else printf("%.4f",min[j]);
	}
	printf("}\n");
	
	printf("{");
	for (int j = 0; j < num_features; j++) {
		if(j<num_features-1) printf("%.4f,",max[j]);
		else printf("%.4f",max[j]);
	}
	printf("}\n");
}