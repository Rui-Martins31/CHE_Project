/*
* João MP Cardoso, Dec. 2023
* Added file_name parameter to functions
* Added option for unsigned char
* Changed input/output calls to read/write from/to files from the beginning. 
*/

#include <stdio.h>
#include <stdlib.h>
#include "types.h"

int read_data_points(char *file_name, int num_features, int num_points, Point *points) 
{
	
	FILE *input_fp=NULL;

	int success=0;

	if ((input_fp = fopen (file_name, "r")) == NULL) {
		printf ("** Error: cannot open %s.\n", file_name);
		exit(1);
	}

	if (num_points <= 0 || num_features <= 0) {
		printf ("** Error: trying to read a negative or zero number of words.\n");
		exit (1);
	}

	for (int i = 0; i < num_points; i++) {
		for (int j = 0; j < num_features; j++) {
			#if DT == 1
			success = fscanf (input_fp, "%lf", &(points[i].features[j]));
			#elif DT == 2
			success = fscanf (input_fp, "%f", &(points[i].features[j]));
			#endif
		}
		int value;
		success = fscanf (input_fp, "%d", &value);
		points[i].classification_id = (CLASS_ID_TYPE) value;
	}  

	fclose(input_fp);
	
	return success;
}


void put(void *value, int data_type, char *msg) {
	int *int_ptr;
	char *char_ptr;
	
	switch (data_type) {
    case 0:
		char_ptr = (char *) value;
		printf("%s %d\n", msg, *char_ptr);
		break;
	case 1:
		int_ptr = (int *) value;
		printf("%s %d\n", msg, *int_ptr);
		break;		
	}
}

