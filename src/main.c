#include "nn.h"
#define LOAD_MNIST_IMPLEMENTAION
#include "nn_data_loader.h"

int main(int argc, char **argv) {
	size_t *sizes = NULL;
	arrsetlen(sizes, 4);
	sizes[0] = 28 * 28;
	sizes[1] = 50;
	sizes[2] = 50;
	sizes[3] = 10;
	Network net = {0};
	if (network_import(&net, "./nn.data") == NN_CODE_FAILURE) {
		network_create(&net, sizes);
	}
	DataEntry *training_set = load_training_set("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", 6e4);
	DataEntry *test_set = load_test_set("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte", 1e4);
	network_SGD(&net, 300, 10, 3.0, training_set, test_set);
	network_export(&net, "./nn.data");
	network_destroy(&net);
	return 0;
}
