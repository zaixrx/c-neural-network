#ifndef NN_H
#define NN_H

#include "nn_math.h"
#include "nn_data_loader.h"

typedef enum {
	NN_CODE_FAILURE  = -1,
	NN_CODE_SUCCESS = 0,
} NN_CODE;

typedef struct {
	size_t *sizes;
	mat_t *weights;
	vec_t *biases;
} Network;

void network_create(Network *net, size_t *sizes);
void network_destroy(Network *net);
void network_SGD(Network *net, size_t epochs, size_t batch_size, double lrate, DataEntry *training_set, DataEntry *test_set);
void network_update_batch(Network *net, DataEntry *batch, double lrate);
void network_backprop(Network *net, DataEntry entry, mat_t *grad_weights, vec_t *grad_biases);
vec_t network_feedforward(Network *net, vec_t input);
int network_test(Network *net, DataEntry entry);
NN_CODE network_export(Network *net, const char *file_path);
NN_CODE network_import(Network *net, const char *file_path);

#endif//NN_H
