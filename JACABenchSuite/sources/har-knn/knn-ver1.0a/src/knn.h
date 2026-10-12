/**
*	k-NN
*
*	Versions
*	- v1.0, January 2025 (from previous versions)
*
*	by João MP Cardoso
*	Email: jmpc@fe.up.pt
*
*	SPeCS, FEUP.DEI, University of Porto, Portugal
*/

#ifndef KNN_H
#define KNN_H

#include "params.h"
#include "types.h"

#ifndef DIMEM
#define DIMEM 0 // 0: not using dynamic memory allocation
#endif

#if DIMEM != 0
#include <stdlib.h>
#endif

#if SPECIALIZED == 1 && K == 3
void copy_3_nearest(BestPoint *dist_points, BestPoint *best_points);

void select_3_nearest(BestPoint *dist_points, int num_points);

void get_3_NN(Point *new_point, Point *known_points, int num_points, BestPoint *best_points, int num_features);

CLASS_ID_TYPE plurality_voting_3(BestPoint *best_points);

CLASS_ID_TYPE knn_classifyinstance_3(Point *new_point, Point *known_points, int num_points, int num_features);

#else
void copy_k_nearest(BestPoint *dist_points, BestPoint *best_points, K_TYPE k);

void select_k_nearest(BestPoint *dist_points, int num_points, K_TYPE k);

void get_k_NN(Point *new_point, Point *known_points, int num_points, BestPoint *best_points, K_TYPE k,  int num_features);

CLASS_ID_TYPE plurality_voting(K_TYPE k, BestPoint *best_points, int num_classes);

CLASS_ID_TYPE knn_classifyinstance(Point *new_point, K_TYPE k, int num_classes, Point *known_points, int num_points, int num_features);
#endif	

#endif
