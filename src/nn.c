#include "nn.h"
#include "math.h"
#include "nn_math.h"
#include "time.h"
#include "stb_ds.h"
#include <stdio.h>

#define TODO(message) do { fprintf(stderr, "%s:%d: TODO: %s\n", __FILE__, __LINE__, message); abort(); } while(0)

static mat_t *new_mat_arr(size_t *sizes) {
	mat_t *arr = NULL;	
	for (size_t l = 1; l < arrlen(sizes); ++l) {
		arrpush(arr, mat_new(sizes[l], sizes[l-1]));
	}
	return arr;
}

static void free_mat_arr(mat_t *arr) {
	for (size_t i = 0; i < arrlen(arr); ++i) {
		mat_destroy(arr[i]);
	}
	arrfree(arr);
}

// allocates array of length: arrlen(sizes[l]) - 1
static vec_t *new_vec_arr(size_t *sizes) {
	vec_t *arr = NULL;	
	for (size_t l = 1; l < arrlen(sizes); ++l) {
		arrpush(arr, vec_new(sizes[l]));
	}
	return arr;
}

static void free_vec_arr(vec_t *arr) {
	for (size_t i = 0; i < arrlen(arr); ++i) {
		vec_destroy(arr[i]);
	}
	arrfree(arr);
}

Network *network_create(size_t *sizes) {
	Network *net = (Network*)malloc(sizeof(Network));
	net->sizes = sizes;
	net->weights = new_mat_arr(sizes);
	net->biases = new_vec_arr(sizes);
	return net;
}

void network_destroy(Network *net) {
	free_mat_arr(net->weights);
	free_vec_arr(net->biases);
	free(net);
}

static void shuffle_set(DataEntry *set) {
	DataEntry temp;
	srand(time(NULL));
	for (int i = 0; i < arrlen(set); ++i) {
		int r = rand() % arrlen(set);
		temp = set[i];
		set[i] = set[r];
		set[r] = temp;
	}
}

static DataEntry **get_batches_from_set(DataEntry *set, size_t batch_size) {
	DataEntry **batches = NULL;
	for (size_t b = 0; b + batch_size < arrlen(set); ++b) {
		DataEntry *batch = NULL;
		for (size_t o = 0; o < batch_size; ++o) {
			arrpush(batch, set[b+o]);
		}
		arrpush(batches, batch);
	}
	return batches;
}

void network_SGD(Network *net, size_t epochs, size_t batch_size, double lrate, DataEntry *training_set, DataEntry *test_set) {
	for (size_t e = 0; e < epochs; ++e) {
		shuffle_set(training_set);
		DataEntry **batches = get_batches_from_set(training_set, batch_size);
		for (size_t b = 0; b < arrlen(batches); ++b) {
			DataEntry *batch = batches[b];
			network_update_batch(net, batch, lrate);
			free(batch);
		}
		arrfree(batches);
		if (test_set) {
			size_t t, ts; // test, test_success
			for (t = 0, ts = 0; t < arrlen(test_set); ++t) {
				ts += network_test(net, test_set[t]);
			}
			printf("Epoch %zu: %zu/%zu\n", e, t, ts);
			continue;
		}
		printf("Epoch %zu\n", e);
	}
}

int network_test(Network *net, DataEntry entry) {
	TODO("network_test");
}

void network_update_batch(Network *net, DataEntry *batch, double lrate) {
	// allocate gradients
	mat_t *grad_weights = new_mat_arr(net->sizes);
	vec_t *grad_biases = new_vec_arr(net->sizes);

	// calculate gradients
	for (size_t b = 0; b < arrlen(batch); ++b) {
		network_backprop(net, batch[b], grad_weights, grad_biases);
	}
	
	TODO("// correct with gradients");
	(void)"net->weights = net->weights - (lrate / arrlen(batch)) * grad_weights";
	(void)"net->biases = net->biases - (lrate / arrlen(batch)) * grad_biases";

	// free gradients
	free_mat_arr(grad_weights);
	free_vec_arr(grad_biases);
}

static void sigmoid(vec_t src, vec_t dst) {
	assert(arrlen(src) == arrlen(dst));
	for (int i = 0; i < arrlen(src); ++i) {
		dst[i] = 1.0 / (1.0 + exp(-src[i]));
	}
}

static void sigmoid_prime(vec_t src, vec_t dst) {
	assert(arrlen(src) == arrlen(dst));
	for (int i = 0; i < arrlen(src); ++i) {
		double e = exp(-src[i]);
		dst[i] = e / ((1 + e) * (1 + e));
	}
}

void vec_print_dims(void *vec) {
	printf("arr.len(): %zu\n", arrlen(vec));
}

void mat_print_dims(void **mat) {
	printf("mat.len(): %zu\n[\n", arrlen(mat));
	for (size_t i = 0; i < arrlen(mat); ++i) {
		vec_print_dims(mat[i]);
	}
	printf("]\n");
}

void network_backprop(Network *net, DataEntry entry, mat_t *grad_weights, vec_t *grad_biases) {
	vec_t *Z = new_vec_arr(net->sizes);
	vec_t *A = new_vec_arr(net->sizes);
	vec_t a = entry.x;
	for (size_t l = 0; l < arrlen(net->sizes)-1; ++l) {
		mat_vec_dot(Z[l], net->weights[l], a);
		vec_operate(Z[l], 1, (VecOp){ ADD, net->biases[l] });
		sigmoid(A[l], Z[l]), a = A[l];
	}
	// ---------- //
	vec_t *D = new_vec_arr(net->sizes);
	mat_print_dims((void**)D);
	for (ssize_t l = arrlen(D) - 1; l >= 0; --l) {
		vec_t zp = NULL;
		arrsetlen(zp, net->sizes[l+1]);
		sigmoid_prime(zp, Z[l]);
		if (l == arrlen(D) - 1) {
			vec_operate(D[l], 2, (VecOp){ LOAD, entry.y }, (VecOp){ SUB, A[l] });
		} else {
			matT_vec_dot(D[l], net->weights[l+1], D[l+1]);
		}
		vec_operate(D[l], 1, (VecOp){ MUL, zp });
		vec_operate(grad_biases[l], 1, (VecOp){ ADD, D[l] }); 
		vecT_vec_dot(grad_weights[l], l > 0 ? A[l-1] : entry.x, D[l]);
		arrfree(zp);
	}
	free_vec_arr(Z);
	free_vec_arr(A);
	free_vec_arr(D);
	TODO("network_backprop");
}
