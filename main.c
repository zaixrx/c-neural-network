// greek dictionary: Σ∂Δδ∇
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <stdlib.h>
#include <stdbool.h>
#include "loadmnist.h"

#define WIDTH (1600)
#define HEIGHT (900)

#define RADIUS (25)
#define HOR_COST (RADIUS * 2 + 250)
#define VER_COST (RADIUS * 2 + 50)
#define LINE_THICKNESS 3

typedef struct {
	size_t *layers;
	float **biases;
	float ***weights;
} Network;

int
rnd() {
	srand(time(NULL));
	return rand();
}

static inline float
rand_uniform() {
    return (rand() + 1.0) / (RAND_MAX + 2.0);
}

// Standard normal via Box-Muller
float
randn() {
    static bool has_spare = 0;
    static float spare;
    if (has_spare) return has_spare = false, spare;
    float u, v;
    u = rand_uniform();
    v = rand_uniform();
    float mag = sqrt(-2.0 * log(u));
    has_spare = true, spare = mag * sin(2.0 * M_PI * v);
    return mag * cos(2.0 * M_PI * v);
}

Network
*new_network(size_t *layers) {
	Network *net = malloc(sizeof(Network));
	net->layers = layers;
	assert(arrlen(net->layers) >= 2);
	for (int l = 0; l < arrlen(layers); ++l) {
		float *biases = NULL;
		for (int i = 0; i < layers[l]; ++i) {
			arrpush(biases, randn());
		}
		arrpush(net->biases, biases);
	}
	for (int l = 1; l < arrlen(layers); ++l) {
		float **matrix = NULL;
		for (int j = 0; j < layers[l]; ++j) {
			float *column = NULL;
			for (int i = 0; i < layers[l-1]; ++i) {
				arrpush(column, randn());
			}
			arrpush(matrix, column);
		}
		arrpush(net->weights, matrix);
	}
	return net;
}

static inline float
sigmoid_prime(float x) {
	float expo = exp(-x);
	return expo / ((1 + expo) * (1 + expo));
}

static inline float
sigmoid(float x) {
	return 1 / (1 + exp(-x));
}

static inline float
*v_sigmoid(float *x) {
	float *a = NULL;
	arrsetlen(a, arrlen(x));
	for (int i = 0; i < arrlen(x); ++i) {
		a[i] = sigmoid(x[i]);
	}
	return a;
}

float
*feed_forward(Network *net, float *data) {
	assert(arrlen(data) == net->layers[0] && "input's length must match input layer's");
	for (int l = 1; l < arrlen(net->layers); ++l) {
		float *output = NULL;
		for (int nn = 0; nn < net->layers[l]; ++nn) {
			float sum = net->biases[l][nn];
			for (int n = 0; n < net->layers[l-1]; ++n) {
				sum += data[n] * net->weights[l][nn][n];
			}
			arrpush(output, sigmoid(sum));
		}
		arrfree(data);
		data = output;
	}
	return data;
}

void
shuffle_set(SetEntry *set) {
	SetEntry temp = {0};
	for (int i = 0; i < arrlen(set); ++i) {
		int j = rnd() % arrlen(set);
		temp = set[i];
		set[i] = set[j];
		set[j] = temp;
	}
}

SetEntry
**generate_mini_batches(SetEntry *set, size_t mini_batch_size) {
	SetEntry **mini_batch_list = NULL;
	for (size_t i = 0, b = 0; i + mini_batch_size < arrlen(set); i += mini_batch_size) {
		SetEntry *mini_batch = NULL;
		for (int j = 0; j < mini_batch_size; ++j) {
			arrpush(mini_batch, set[i + j]);
		}
		// printf("%zu: %zu\n", i, arrlen(mini_batch));
		arrpush(mini_batch_list, mini_batch);
	}
	return mini_batch_list;
}

void
backprop(Network *net, SetEntry entry, float ***nabla_w, float **nabla_b) {
        float **A = NULL, **Z = NULL, *a, *z; // @heap_allocated
	size_t L = arrlen(net->layers);
	arrpush(Z, z = NULL);
	arrpush(A, a = entry.x);
	for (int l = 1; l < L; ++l) {
		z = NULL;
		for (int j = 0; j < net->layers[l]; ++j) {
			float sum = net->biases[l][j];
			for (int k = 0; k < net->layers[l-1]; ++k) {
				sum += a[k] * net->weights[l][j][k];
			}
			arrpush(z, sum);
		}
		arrpush(Z, z);
		arrpush(A, a = v_sigmoid(z));
	}
	float *deltas = NULL;
	for (int l = L - 1; l >= 1; --l) {
		if (l == L - 1)
			for (int j = 0; j < net->layers[l]; ++j) {
				float y = (int)entry.y == j + 1;
				float delta = (y - A[l][j]) * sigmoid_prime(Z[l][j]);
				arrpush(deltas, delta);
			}
		else {
			float *new_deltas = NULL;
			for (int j = 0; j < net->layers[l+1]; ++j) {
				float delta = 0.0f;
				for (int k = 0; k < net->layers[l]; ++k) {
					delta += deltas[j] * net->weights[l+1][j][k];
				}
				delta *= sigmoid_prime(Z[l][j]);
				arrpush(new_deltas, delta);
			}
			arrfree(deltas);
			deltas = new_deltas;
		}
		for (int j = 0; j < net->layers[l]; ++j) {
			for (int k = 0; k < net->layers[l-1]; ++k) {
				nabla_w[l][j][k] += deltas[j] * A[l][k];
			}
			nabla_b[l][j] += deltas[j];
		}
	}
	arrfree(deltas);
	// narrfree(A, 2);
	// narrfree(Z, 2);
}

void
update_mini_batch(Network *net, SetEntry *mini_batch_set, float eta) {
	float **nabla_b = NULL, ***nabla_w = NULL;
	// initialization
	arrsetlen(nabla_b, arrlen(net->layers));
	arrsetlen(nabla_w, arrlen(net->layers));
	for (int l = 1; l < arrlen(net->layers); ++l) {
		arrsetlen(nabla_b[l], net->layers[l]);
		arrsetlen(nabla_w[l], net->layers[l]);
		for (int j = 0; j < net->layers[l]; ++j) {
			arrsetlen(nabla_w[l][j], net->layers[l-1]);
			for (int k = 0; k < net->layers[l-1]; ++k) {
				nabla_w[l][j][k] = 0;
			}
			nabla_b[l][j] = 0;
		}
	}
	// backpropagation
	for (int b = 0; b < arrlen(mini_batch_set); ++b) {
		backprop(net, mini_batch_set[b], nabla_w, nabla_b);
	}
	// update weights and biases
	for (int l = 1; l < arrlen(net->layers); ++l) {
		for (int j = 0; j < net->layers[l]; ++j) {
			for (int k = 0; k < net->layers[l-1]; ++k) {
				net->weights[l][j][k] += -eta / arrlen(mini_batch_set) * nabla_w[l][j][k];
			}
			net->biases[l][j] += -eta / arrlen(mini_batch_set) * nabla_b[l][j];
		}
	}
	// narrfree(3, nabla_w);
	// narrfree(2, nabla_b);
}

void
SGD(Network *net, SetEntry *training_data, size_t epochs, size_t mini_batch_size, float eta, SetEntry *test_data) {
	for (size_t e = 0; e < epochs; ++e) {
		shuffle_set(training_data);
		SetEntry **mini_batches = generate_mini_batches(training_data, mini_batch_size);
		for (int b = 0; b < arrlen(mini_batches); ++b) {
			update_mini_batch(net, mini_batches[b], eta);
			arrfree(mini_batches[b]);
		}
		arrfree(mini_batches);
		printf("Epoch %zu / %zu\n", e, epochs);
		if (test_data == NULL) continue;
		size_t sucess, total;
		for (sucess = 0, total = 0; total < arrlen(test_data); ++total) {
			float *res = feed_forward(net, test_data[total].x);
			size_t max = 0, expect = (int)test_data[total].y - 1;
			for (size_t j = 0; j < arrlen(res); ++j) { if (res[total] > res[max]) max = total; }
			if (max == expect) ++sucess;
			arrfree(res);
		}
		printf("Score: %zu / %zu\n", sucess, total);
	}
}

int
main(void) {
	Network *net = NULL;
	size_t *layers = NULL;
	arrpush(layers, 5);
	arrpush(layers, 7);
	arrpush(layers, 3);
	arrpush(layers, 8);
	arrpush(layers, 1);
	net = new_network(layers);
	
	SetEntry *training_set = get_training_set();
	SetEntry *test_set = get_test_set();
	SGD(net, training_set, 10, 30, 3.0F, test_set);

    	return 0;
}
