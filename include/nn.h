#ifndef NN_H
#define NN_H

#include "nn_math.h"
#include "nn_data_loader.h"

typedef enum {
	NN_CODE_FAILURE  = -1,
	NN_CODE_SUCCESS = 0,
} NN_CODE;

typedef enum {
	NN_LINEAR,
	NN_ReLU,
	NN_SIGMOID,
} NN_Activation;

typedef struct {
	NN_Activation activation;
	size_t *sizes;
	mat_t *weights;
	vec_t *biases;
} Network;

// variadic parameter represents network.sizes
void network_create(Network *net, size_t layers_count, ...);
void network_destroy(Network *net);
void network_SGD(Network *net, size_t batch_size, double lrate, DataEntry *training_set);
void network_update_batch(Network *net, DataEntry *batch, double lrate);
void network_backprop(Network *net, DataEntry entry, mat_t *grad_weights, vec_t *grad_biases);
vec_t network_feedforward(Network *net, vec_t input);

double network_cost(Network *net, DataEntry entry);
NN_CODE network_export(Network *net, const char *file_path);
NN_CODE network_import(Network *net, const char *file_path);

#endif//NN_H
