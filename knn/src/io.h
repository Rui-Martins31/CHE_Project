#ifndef HAR_IO_H
#define HAR_IO_H

/**
* Read data from .dat files.
*/

int read_data_points(char *file_name, int num_features, int num_points, Point *points);				
			
void put(void *value, int data_type, char *msg);
			
#endif
