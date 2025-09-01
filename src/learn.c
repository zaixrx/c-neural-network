#include "nn.h"
#include <assert.h>
#include <stdio.h>
#define LOAD_MNIST_IMPLEMENTAION
#include "nn_data_loader.h"

typedef struct {
	int offset;
	int count;
	char **items;
} Args;

Args new_args(int argc, char **argv) {
	assert(argc >= 1 && "new_args: program_name required");
	return (Args){
		.offset = 0,
		.count = argc,
		.items = argv,
	};
}

char *args_str(Args *args) {
	assert(args->offset < args->count && "Args::args_str");
	return args->items[args->offset++];
}

uint32_t args_uint32(Args *args) {
	char *num = args_str(args);
	return (uint32_t)atoi(num);
}

int main(int argc, char **argv) {
	Args args = new_args(argc, argv);
	char *program_name = args_str(&args);
	if (args.count != 2) {
		fprintf(stderr, "usage: %s <epochs_count>\n", program_name);
		return 1;
	}
	uint32_t epochs_count = args_uint32(&args);

	Network net = {0};
	if (network_import(&net, "./nn.data") == NN_CODE_FAILURE) {
		network_create(&net, 3, 28*28, 50, 10);
	}

	DataEntry *training_set = load_training_set("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", 6e4);
	DataEntry *test_set = load_test_set("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte", 1e4);

	network_SGD(&net, epochs_count, 10, 3.0, training_set, test_set);
	network_export(&net, "./nn.data");
	network_destroy(&net);

	return 0;
}
