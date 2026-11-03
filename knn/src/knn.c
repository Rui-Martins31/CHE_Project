/**
*	k-NN
*
*	Versions
*	- v1.0, January 2025 (from previous versions)
*
*	by Jo�o MP Cardoso
*	Email: jmpc@fe.up.pt
*
*	SPeCS, FEUP.DEI, University of Porto, Portugal
*/

#include <math.h>
#include <omp.h>
#include "knn.h"
#include "params.h"

#if SPECIALIZED == 1 && K == 3
#define SWAP(a,b,c) c = a; a = b; b = c;  

/**
*  Get the 3 nearest points.
*  This version stores the 3 nearest points in the first 3 positions of dist_point
*/
void select_3_nearest(BestPoint *dist_points, int num_points) {

    DATA_TYPE md1, md2, md3, mdaux;
    int di1, di2, di3, diaux;

	md1 = dist_points[0].distance;
	di1 = 0;
	md2 = dist_points[1].distance;
	di2 = 1;
	md3 = dist_points[2].distance;
	di3 = 2;
	
	if(md1 > md2) {
		SWAP(md1, md2, mdaux);
		SWAP(di1, di2, diaux);
	} 
	if(md2 > md3) {
		SWAP(md2, md3, mdaux);
		SWAP(di2, di3, diaux);
	}
	if(md1 > md2) {
		SWAP(md1, md2, mdaux);
		SWAP(di1, di2, diaux);
	}
		
	for(int i = 3; i < num_points; i++) {
		mdaux = dist_points[i].distance;
		diaux = i;
		
		if(mdaux < md1) {
			md3 = md2; di3 = di2;
			md2 = md1; di2 = di1;
			md1 = mdaux; di1 = diaux;
		} else if(mdaux < md2) {
			md3 = md2; di3 = di2;
			md2 = mdaux; di2 = diaux;
		} else if(mdaux < md3) {
			md3 = mdaux; di3 = diaux;
		}
	}
            
	dist_points[0].distance = dist_points[di1].distance;
	dist_points[0].classification_id = dist_points[di1].classification_id;
	
	dist_points[1].distance = dist_points[di2].distance;
	dist_points[1].classification_id = dist_points[di2].classification_id;
	
	dist_points[2].distance = dist_points[di3].distance;
	dist_points[2].classification_id = dist_points[di3].classification_id;
}

/*
*	Classify using the 3 nearest neighbors identified by the get_3_NN
*	function. The classification uses plurality voting.
*
*	Note: it assumes that classes are identified from 0 to num_classes - 1.
*/
CLASS_ID_TYPE plurality_voting_3(BestPoint *best_points) {

	CLASS_ID_TYPE ids0 = best_points[0].classification_id;
    CLASS_ID_TYPE ids1 = best_points[1].classification_id;
    CLASS_ID_TYPE ids2 = best_points[2].classification_id;
	
	CLASS_ID_TYPE classification_id = ids0; // assuming they are sorted then the first one is the class
		
    if(ids0 == ids2)
        classification_id = ids0;
    else if(ids1 == ids2)
        classification_id = ids1;
	
    return classification_id;
}

void copy_3_nearest(BestPoint *dist_points, BestPoint *best_points) {
    best_points[0].classification_id = dist_points[0].classification_id;
    best_points[0].distance = dist_points[0].distance;
	
    best_points[1].classification_id = dist_points[1].classification_id;
    best_points[1].distance = dist_points[1].distance;
	
    best_points[2].classification_id = dist_points[2].classification_id;
    best_points[2].distance = dist_points[2].distance;
}

static int nearer3(DATA_TYPE d_a, int i_a, DATA_TYPE d_b, int i_b) {
	return d_a < d_b || (d_a == d_b && i_a < i_b);
}

static void insert3(DATA_TYPE distance, int index, CLASS_ID_TYPE cid,
	DATA_TYPE *md1, DATA_TYPE *md2, DATA_TYPE *md3,
	int *di1, int *di2, int *di3,
	CLASS_ID_TYPE *c1, CLASS_ID_TYPE *c2, CLASS_ID_TYPE *c3) {
	if (index < 0)
		return;
	if (nearer3(distance, index, *md1, *di1)) {
		*md3 = *md2; *di3 = *di2; *c3 = *c2;
		*md2 = *md1; *di2 = *di1; *c2 = *c1;
		*md1 = distance; *di1 = index; *c1 = cid;
	} else if (nearer3(distance, index, *md2, *di2)) {
		*md3 = *md2; *di3 = *di2; *c3 = *c2;
		*md2 = distance; *di2 = index; *c2 = cid;
	} else if (nearer3(distance, index, *md3, *di3)) {
		*md3 = distance; *di3 = index; *c3 = cid;
	}
}

void get_3_NN(Point *new_point, Point *known_points, int num_points,
	BestPoint *best_points, int num_features) {

	// Running top-3, split across threads on the training points.
	// Each thread keeps its own 3. The merge uses (distance, index):
	// strict <, then the smaller training index, matching select_3_nearest.
	DATA_TYPE md1 = MAX_FP_VAL, md2 = MAX_FP_VAL, md3 = MAX_FP_VAL;
	int di1 = -1, di2 = -1, di3 = -1;
	CLASS_ID_TYPE c1 = 0, c2 = 0, c3 = 0;

	#pragma omp parallel
	{
		DATA_TYPE p1 = MAX_FP_VAL, p2 = MAX_FP_VAL, p3 = MAX_FP_VAL;
		int i1 = -1, i2 = -1, i3 = -1;
		CLASS_ID_TYPE k1 = 0, k2 = 0, k3 = 0;

		#pragma omp for nowait schedule(static)
		for (int i = 0; i < num_points; i++) {
			DATA_TYPE distance = (DATA_TYPE) 0.0;

			#if DIST_METHOD == 1
			for (int j = 0; j < num_features; j++) {
				DATA_TYPE diff = (DATA_TYPE) new_point->features[j] - (DATA_TYPE) known_points[i].features[j];
				distance += diff * diff;
			}
			#if USE_SQRT == 1
				#if MATH_TYPE == 1 && DT == 2
					distance = sqrtf(distance);
				#else
					distance = sqrt(distance);
				#endif
			#endif
			#elif DIST_METHOD == 2
			for (int j = 0; j < num_features; j++) {
				#if MATH_TYPE == 1 && DT == 2
				DATA_TYPE absdiff = fabsf((DATA_TYPE) new_point->features[j] - (DATA_TYPE) known_points[i].features[j]);
				#else
				DATA_TYPE absdiff = fabs((DATA_TYPE) new_point->features[j] - (DATA_TYPE) known_points[i].features[j]);
				#endif
				distance += absdiff;
			}
			#endif

			CLASS_ID_TYPE cid = known_points[i].classification_id;
			if (distance < p1) {
				p3 = p2; i3 = i2; k3 = k2;
				p2 = p1; i2 = i1; k2 = k1;
				p1 = distance; i1 = i; k1 = cid;
			} else if (distance < p2) {
				p3 = p2; i3 = i2; k3 = k2;
				p2 = distance; i2 = i; k2 = cid;
			} else if (distance < p3) {
				p3 = distance; i3 = i; k3 = cid;
			}
		}

		#pragma omp critical
		{
			insert3(p1, i1, k1, &md1, &md2, &md3, &di1, &di2, &di3, &c1, &c2, &c3);
			insert3(p2, i2, k2, &md1, &md2, &md3, &di1, &di2, &di3, &c1, &c2, &c3);
			insert3(p3, i3, k3, &md1, &md2, &md3, &di1, &di2, &di3, &c1, &c2, &c3);
		}
	}

	best_points[0].distance = md1;
	best_points[0].classification_id = c1;
	best_points[1].distance = md2;
	best_points[1].classification_id = c2;
	best_points[2].distance = md3;
	best_points[2].classification_id = c3;
}

/*
* Classify nbatch points in one parallel pass over the training set.
* Query features are contiguous per point. The training point is reused
* across the batch. Classes are written in the same order as new_points.
*/
void knn_classify_batch_3(Point *new_points, int nbatch, Point *known_points,
	int num_points, int num_features, CLASS_ID_TYPE *classes) {
	(void)num_features;

	DATA_TYPE qfeat[KNN_BATCH][NUM_FEATURES];
	for (int t = 0; t < nbatch; t++) {
		for (int j = 0; j < NUM_FEATURES; j++)
			qfeat[t][j] = new_points[t].features[j];
	}

	DATA_TYPE md1[KNN_BATCH], md2[KNN_BATCH], md3[KNN_BATCH];
	int di1[KNN_BATCH], di2[KNN_BATCH], di3[KNN_BATCH];
	CLASS_ID_TYPE c1[KNN_BATCH], c2[KNN_BATCH], c3[KNN_BATCH];
	for (int t = 0; t < nbatch; t++) {
		md1[t] = md2[t] = md3[t] = MAX_FP_VAL;
		di1[t] = di2[t] = di3[t] = -1;
		c1[t] = c2[t] = c3[t] = 0;
	}

	#pragma omp parallel
	{
		DATA_TYPE p1[KNN_BATCH], p2[KNN_BATCH], p3[KNN_BATCH];
		int i1[KNN_BATCH], i2[KNN_BATCH], i3[KNN_BATCH];
		CLASS_ID_TYPE k1[KNN_BATCH], k2[KNN_BATCH], k3[KNN_BATCH];
		for (int t = 0; t < nbatch; t++) {
			p1[t] = p2[t] = p3[t] = MAX_FP_VAL;
			i1[t] = i2[t] = i3[t] = -1;
			k1[t] = k2[t] = k3[t] = 0;
		}

		#pragma omp for nowait schedule(static)
		for (int i = 0; i < num_points; i++) {
			DATA_TYPE train[NUM_FEATURES];
			for (int j = 0; j < NUM_FEATURES; j++)
				train[j] = known_points[i].features[j];
			CLASS_ID_TYPE cid = known_points[i].classification_id;

			for (int t = 0; t < nbatch; t++) {
				DATA_TYPE distance = (DATA_TYPE) 0.0;
				#if DIST_METHOD == 1
				for (int j = 0; j < NUM_FEATURES; j++) {
					DATA_TYPE diff = qfeat[t][j] - train[j];
					distance += diff * diff;
				}
				#if USE_SQRT == 1
					#if MATH_TYPE == 1 && DT == 2
						distance = sqrtf(distance);
					#else
						distance = sqrt(distance);
					#endif
				#endif
				#elif DIST_METHOD == 2
				for (int j = 0; j < NUM_FEATURES; j++) {
					#if MATH_TYPE == 1 && DT == 2
					DATA_TYPE absdiff = fabsf(qfeat[t][j] - train[j]);
					#else
					DATA_TYPE absdiff = fabs(qfeat[t][j] - train[j]);
					#endif
					distance += absdiff;
				}
				#endif

				if (distance < p1[t]) {
					p3[t] = p2[t]; i3[t] = i2[t]; k3[t] = k2[t];
					p2[t] = p1[t]; i2[t] = i1[t]; k2[t] = k1[t];
					p1[t] = distance; i1[t] = i; k1[t] = cid;
				} else if (distance < p2[t]) {
					p3[t] = p2[t]; i3[t] = i2[t]; k3[t] = k2[t];
					p2[t] = distance; i2[t] = i; k2[t] = cid;
				} else if (distance < p3[t]) {
					p3[t] = distance; i3[t] = i; k3[t] = cid;
				}
			}
		}

		#pragma omp critical
		{
			for (int t = 0; t < nbatch; t++) {
				insert3(p1[t], i1[t], k1[t], &md1[t], &md2[t], &md3[t], &di1[t], &di2[t], &di3[t], &c1[t], &c2[t], &c3[t]);
				insert3(p2[t], i2[t], k2[t], &md1[t], &md2[t], &md3[t], &di1[t], &di2[t], &di3[t], &c1[t], &c2[t], &c3[t]);
				insert3(p3[t], i3[t], k3[t], &md1[t], &md2[t], &md3[t], &di1[t], &di2[t], &di3[t], &c1[t], &c2[t], &c3[t]);
			}
		}
	}

	for (int t = 0; t < nbatch; t++) {
		BestPoint best_points[3];
		best_points[0].distance = md1[t];
		best_points[0].classification_id = c1[t];
		best_points[1].distance = md2[t];
		best_points[1].classification_id = c2[t];
		best_points[2].distance = md3[t];
		best_points[2].classification_id = c3[t];
		classes[t] = plurality_voting_3(best_points);
	}
}

/*
* Classify a given Point (instance) with a code version specialized for k=3.
* It returns the classified class ID.
*/
CLASS_ID_TYPE knn_classifyinstance_3(Point *new_point, Point *known_points, int num_points, int num_features) {

	//BestPoint *best_points = (BestPoint *) calloc(k, sizeof(BestPoint)) ;
    BestPoint best_points[3]; // Array with the k nearest points to the Point to classify

    // calculate the distances of the new point to each of the known points and get
    // the k nearest points
    get_3_NN(new_point, known_points, num_points, best_points, num_features);

	// use plurality voting of 3 classes to return the class inferred for the new point
	CLASS_ID_TYPE classID = plurality_voting_3(best_points);
	
	return classID;
}

#else
/**
*  Copy the top k nearest points (first k elements of dist_points)
*  to a data structure (best_points) with k points
*/
void copy_k_nearest(BestPoint *dist_points, BestPoint *best_points, K_TYPE k) {

    for(int i = 0; i < k; i++) {   // we only need the top k minimum distances
       best_points[i].classification_id = dist_points[i].classification_id;
       best_points[i].distance = dist_points[i].distance;
    }

}

/**
*  Get the k nearest points.
*  This version stores the k nearest points in the first k positions of dist_point
*/
void select_k_nearest(BestPoint *dist_points, int num_points, K_TYPE k) {

    DATA_TYPE min_distance, distance_i;
    CLASS_ID_TYPE class_id_1;
    int index;

    for(int i = 0; i < k; i++) {  // we only need the top k minimum distances
		min_distance = dist_points[i].distance;
		index = i;
		for(int j = i+1; j < num_points; j++) {
            if(dist_points[j].distance < min_distance) {
                min_distance = dist_points[j].distance;
                index = j;
            }
		}
		if(index != i) { //swap
			distance_i = dist_points[index].distance;
			class_id_1 = dist_points[index].classification_id;

			dist_points[index].distance = dist_points[i].distance;
			dist_points[index].classification_id = dist_points[i].classification_id;

			dist_points[i].distance = distance_i;
			dist_points[i].classification_id = class_id_1;
		}
    }
}


/*
* Main kNN function.
* It calculates the distances and calculates the nearest k points.
*/
static void merge_neighbor(DATA_TYPE distance, int index, CLASS_ID_TYPE cid, int k,
	DATA_TYPE *best_d, int *best_i, CLASS_ID_TYPE *best_c) {
	if (index < 0)
		return;
	if (!(distance < best_d[k - 1] || (distance == best_d[k - 1] && index < best_i[k - 1])))
		return;
	int slot = k - 1;
	while (slot > 0 && (distance < best_d[slot - 1] ||
			(distance == best_d[slot - 1] && index < best_i[slot - 1]))) {
		best_d[slot] = best_d[slot - 1];
		best_i[slot] = best_i[slot - 1];
		best_c[slot] = best_c[slot - 1];
		slot--;
	}
	best_d[slot] = distance;
	best_i[slot] = index;
	best_c[slot] = cid;
}

void get_k_NN(Point *new_point, Point *known_points, int num_points,
	BestPoint *best_points, K_TYPE k,  int num_features) {

	// Running top-k, split across threads on the training points.
	// Merge key is (distance, index), matching select_k_nearest.
	DATA_TYPE best_d[k];
	int best_i[k];
	CLASS_ID_TYPE best_c[k];
	for (int t = 0; t < k; t++) {
		best_d[t] = MAX_FP_VAL;
		best_i[t] = -1;
		best_c[t] = 0;
	}

	#pragma omp parallel
	{
		DATA_TYPE pd[k];
		int pi[k];
		CLASS_ID_TYPE pc[k];
		for (int t = 0; t < k; t++) {
			pd[t] = MAX_FP_VAL;
			pi[t] = -1;
			pc[t] = 0;
		}

		#pragma omp for nowait schedule(static)
		for (int i = 0; i < num_points; i++) {
			DATA_TYPE distance = (DATA_TYPE) 0.0;

			#if DIST_METHOD == 1
			for (int j = 0; j < num_features; j++) {
				DATA_TYPE diff = (DATA_TYPE) new_point->features[j] - (DATA_TYPE) known_points[i].features[j];
				distance += diff * diff;
			}
			#if USE_SQRT == 1
				#if MATH_TYPE == 1 && DT == 2
					distance = sqrtf(distance);
				#else
					distance = sqrt(distance);
				#endif
			#endif
			#elif DIST_METHOD == 2
			for (int j = 0; j < num_features; j++) {
				#if MATH_TYPE == 1 && DT == 2
				DATA_TYPE absdiff = fabsf((DATA_TYPE) new_point->features[j] - (DATA_TYPE) known_points[i].features[j]);
				#else
				DATA_TYPE absdiff = fabs((DATA_TYPE) new_point->features[j] - (DATA_TYPE) known_points[i].features[j]);
				#endif
				distance += absdiff;
			}
			#endif

			if (distance < pd[k - 1]) {
				int slot = k - 1;
				while (slot > 0 && distance < pd[slot - 1]) {
					pd[slot] = pd[slot - 1];
					pi[slot] = pi[slot - 1];
					pc[slot] = pc[slot - 1];
					slot--;
				}
				pd[slot] = distance;
				pi[slot] = i;
				pc[slot] = known_points[i].classification_id;
			}
		}

		#pragma omp critical
		{
			for (int t = 0; t < k; t++)
				merge_neighbor(pd[t], pi[t], pc[t], k, best_d, best_i, best_c);
		}
	}

	for (int t = 0; t < k; t++) {
		best_points[t].distance = best_d[t];
		best_points[t].classification_id = best_c[t];
	}
}

/*
*	Classify using the k nearest neighbors identified by the get_k_NN
*	function. The classification uses plurality voting.
*
*	Note: it assumes that classes are identified from 0 to num_classes - 1.
*/
CLASS_ID_TYPE plurality_voting(K_TYPE k, BestPoint *best_points, int num_classes) {

	#if DIMEM == 0
	K_TYPE histogram[num_classes];  // maximum equals the value of k;
	//initialize the histogram
    for (int i = 0; i < num_classes; i++) {
        histogram[i] = 0;
    }
    #else
	K_TYPE *histogram = (K_TYPE *) calloc(NUM_CLASSES, sizeof(CLASS_ID_TYPE)) ;
	#endif

    // build the histogram
    for (int i = 0; i < k; i++) {
        BestPoint p = best_points[i];
        histogram[(int) p.classification_id] += 1;
    }
	
	CLASS_ID_TYPE classification_id = best_points[0].classification_id;
    K_TYPE max = 1; // maximum is k
    for (int i = 0; i < num_classes; i++) {

        if (histogram[i] > max) {
            max = histogram[i];
            classification_id = (CLASS_ID_TYPE) i;
        }
    }

	#if DIMEM != 0
	free(histogram);
	#endif
	
    return classification_id;
}

/*
* Classify nbatch points in one parallel pass over the training set.
* Query features are contiguous per point. The training point is reused
* across the batch. Classes are written in the same order as new_points.
*/
void knn_classify_batch(Point *new_points, int nbatch, K_TYPE k, int num_classes,
	Point *known_points, int num_points, int num_features, CLASS_ID_TYPE *classes) {
	(void)num_features;

	DATA_TYPE qfeat[KNN_BATCH][NUM_FEATURES];
	for (int t = 0; t < nbatch; t++) {
		for (int j = 0; j < NUM_FEATURES; j++)
			qfeat[t][j] = new_points[t].features[j];
	}

	DATA_TYPE best_d[KNN_BATCH][k];
	int best_i[KNN_BATCH][k];
	CLASS_ID_TYPE best_c[KNN_BATCH][k];
	for (int t = 0; t < nbatch; t++) {
		for (int s = 0; s < k; s++) {
			best_d[t][s] = MAX_FP_VAL;
			best_i[t][s] = -1;
			best_c[t][s] = 0;
		}
	}

	#pragma omp parallel
	{
		DATA_TYPE pd[KNN_BATCH][k];
		int pi[KNN_BATCH][k];
		CLASS_ID_TYPE pc[KNN_BATCH][k];
		for (int t = 0; t < nbatch; t++) {
			for (int s = 0; s < k; s++) {
				pd[t][s] = MAX_FP_VAL;
				pi[t][s] = -1;
				pc[t][s] = 0;
			}
		}

		#pragma omp for nowait schedule(static)
		for (int i = 0; i < num_points; i++) {
			DATA_TYPE train[NUM_FEATURES];
			for (int j = 0; j < NUM_FEATURES; j++)
				train[j] = known_points[i].features[j];
			CLASS_ID_TYPE cid = known_points[i].classification_id;

			for (int t = 0; t < nbatch; t++) {
				DATA_TYPE distance = (DATA_TYPE) 0.0;
				#if DIST_METHOD == 1
				for (int j = 0; j < NUM_FEATURES; j++) {
					DATA_TYPE diff = qfeat[t][j] - train[j];
					distance += diff * diff;
				}
				#if USE_SQRT == 1
					#if MATH_TYPE == 1 && DT == 2
						distance = sqrtf(distance);
					#else
						distance = sqrt(distance);
					#endif
				#endif
				#elif DIST_METHOD == 2
				for (int j = 0; j < NUM_FEATURES; j++) {
					#if MATH_TYPE == 1 && DT == 2
					DATA_TYPE absdiff = fabsf(qfeat[t][j] - train[j]);
					#else
					DATA_TYPE absdiff = fabs(qfeat[t][j] - train[j]);
					#endif
					distance += absdiff;
				}
				#endif

				if (distance < pd[t][k - 1]) {
					int slot = k - 1;
					while (slot > 0 && distance < pd[t][slot - 1]) {
						pd[t][slot] = pd[t][slot - 1];
						pi[t][slot] = pi[t][slot - 1];
						pc[t][slot] = pc[t][slot - 1];
						slot--;
					}
					pd[t][slot] = distance;
					pi[t][slot] = i;
					pc[t][slot] = cid;
				}
			}
		}

		#pragma omp critical
		{
			for (int t = 0; t < nbatch; t++) {
				for (int s = 0; s < k; s++)
					merge_neighbor(pd[t][s], pi[t][s], pc[t][s], k, best_d[t], best_i[t], best_c[t]);
			}
		}
	}

	for (int t = 0; t < nbatch; t++) {
		BestPoint best_points[k];
		for (int s = 0; s < k; s++) {
			best_points[s].distance = best_d[t][s];
			best_points[s].classification_id = best_c[t][s];
		}
		classes[t] = plurality_voting(k, best_points, num_classes);
	}
}

/*
* Classify a given Point (instance).
* It returns the classified class ID.
*/
CLASS_ID_TYPE knn_classifyinstance(Point *new_point, K_TYPE k, int num_classes, Point *known_points, int num_points, int num_features) {

	#if DIMEM != 0
	BestPoint *best_points = (BestPoint *) calloc(k, sizeof(BestPoint)) ;
    #else
	BestPoint best_points[k]; // Array with the k nearest points to the Point to classify
	#endif
	
    // calculate the distances of the new point to each of the known points and get
    // the k nearest points
    get_k_NN(new_point, known_points, num_points, best_points, k, num_features);

	// use plurality voting to return the class inferred for the new point
	CLASS_ID_TYPE classID = plurality_voting(k, best_points, num_classes);

	// content of the k best
	//for (int i = 0; i < k; i++) {
		//printf("ID = %d | distance = %e\n",best_points[i].classification_id, best_points[i].distance);
    //}
	#if DIMEM != 0
	free(best_points);
	#endif
	
	return classID;
}

#endif
