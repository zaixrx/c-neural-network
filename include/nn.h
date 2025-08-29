#ifndef NN_H
#define NN_H

#include "nn_math.h"
#include "nn_data_loader.h"

typedef struct {
	size_t *sizes;
	mat_t *weights;
	vec_t *biases;
} Network;

Network *network_create(size_t *sizes);
void network_destroy(Network *net);
// lrate: learning rate
void network_SGD(Network *net, size_t epochs, size_t batch_size, double lrate, DataEntry *training_set, DataEntry *test_set);
void network_update_batch(Network *net, DataEntry *batch, double lrate);
void network_backprop(Network *net, DataEntry entry, mat_t *grad_weights, vec_t *grad_biases);
int network_test(Network *net, DataEntry entry);

#endif//NN_H
