#define NN_MATH_IMPLEMENTATION
#include "nn.h"
#define LOAD_MNIST_IMPLEMENTAION
#include "nn_data_loader.h"
#define STB_DS_IMPLEMENTATION
#include "stb_ds.h"

int main(void) {
	size_t *sizes = NULL;
	arrsetlen(sizes, 3);
	sizes[0] = 28 * 28;
	sizes[1] = 30;
	sizes[2] = 10;
	Network *net = network_create(sizes);
	DataEntry *training_set = load_training_set();
	DataEntry *test_set= load_training_set();
	network_SGD(net, 10, 50, 3.0, training_set, test_set);
	network_destroy(net);
}
