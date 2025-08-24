// greek dictionary: Σ∂Δδ∇

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include "include/raylib.h"
#include "loadmnist.h"

#define WIDTH (1600)
#define HEIGHT (900)

#define RADIUS (25)
#define HOR_COST (RADIUS * 2 + 250)
#define VER_COST (RADIUS * 2 + 50)
#define LINE_THICKNESS 3

int rnd() {
	srand(time(NULL));
	return rand();
}

typedef struct {
	size_t *layers;
	float **biases;
	float ***weights;
} Network;

Network *new_network(size_t *layers) {
	Network *net = malloc(sizeof(Network));
	net->layers = layers;
	assert(arrlen(net->layers) >= 2);
	for (int l = 0; l < arrlen(layers); ++l) {
		float *biases = NULL;
		for (int i = 0; i < layers[l]; ++i) {
			arrpush(biases, 69); // TODO: must be random
		}
		arrpush(net->biases, biases);
	}
	for (int l = 1; l < arrlen(layers); ++l) {
		float **matrix = NULL;
		for (int j = 0; j < layers[l]; ++j) {
			float *column = NULL;
			for (int i = 0; i < layers[l-1]; ++i) {
				arrpush(column, 69);
			}
			arrpush(matrix, column);
		}
		arrpush(net->weights, matrix);
	}
	return net;
}

void
DrawNeurons(Network *net) {
	static Color rgb[] = { RED, GREEN, BLUE };

	clock_t start = clock();
	float x = (WIDTH - (arrlen(net->layers) - 1) * HOR_COST) / 2.0f;
	DrawCircle(x, HEIGHT >> 1, RADIUS, RED);
	for (int l = 0; l < arrlen(net->layers); ++l) {
		float y = (HEIGHT - (net->layers[l] - 1) * VER_COST) / 2.0f;
		for (int n = 0; n < net->layers[l]; ++n) {
			DrawCircle(x, y, RADIUS, WHITE);
			if (l + 1 < arrlen(net->layers)) {
				float ny = (HEIGHT - (net->layers[l+1]-1) * VER_COST) / 2.0f;
				for (int nn = 0; nn < net->layers[l+1]; ++nn) {
					DrawLineEx(
						(Vector2){x + RADIUS, y},
						(Vector2){x + HOR_COST - RADIUS, ny},
						LINE_THICKNESS,
						rgb[l % 3]
					);
					ny += VER_COST;
				}
			}
			y += VER_COST;
		}
		x += HOR_COST;
	}
	DrawCircle(x, HEIGHT >> 1, RADIUS, RED);
	clock_t end = clock();

	printf("DrawNeurons :: took %fµs\n", (float)(end - start));
}

inline float
sigmoid_prime(float x) {
	float expo = exp(-x);
	return expo / ((1 + expo) * (1 + expo));
}

inline float
sigmoid(float x) {
	return 1 / (1 + exp(-x));
}

inline float
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
**generate_mini_batches(SetEntry *data, size_t mini_batch_size) {
	SetEntry **mini_batches = NULL;
	for (int i = 0; i < arrlen(data); i += mini_batch_size) {
		SetEntry *mini_batch = NULL;
		arrsetlen(mini_batch, mini_batch_size);
		memcpy(mini_batch, data + i, mini_batch_size);
		arrpush(mini_batches, mini_batch);
	}
	return NULL;
}

/*
foreach test input we want to calculate the change of each weight and bias:
	∇C = (∂C/∂W, ∂C/∂B)

for l > 1: δl = ∂C/∂Z(l)
1) δL = hadamard(y - aL, sigma_prime(zL))
2) for l < L: δ(l) = hadamard(dot(W(l+1), δ(l+1)), sigma_prime(Z(l)))

-- finally --
given Z(l) = dot(W(l), A(l-1)) + B(l) we conclude:
     	∂C/∂W(l) = hadamard(δ(l), A(l-1)) ... (1)
also:
     	∂C/∂B(l) = δ(l) ... (2)
hence:
   	∇C(l) = (∂C/∂W(l), ∂C/∂B(l))
do that foreach l > 1 and you get ∇C for a specific test input.
*/
void
backprop(Network *net, SetEntry *batch, float ***nabla_w, float **nabla_b) {
	float *x = batch.x, *y = batch.y;
	size_t L = arrlen(net->layers);
        float **A = NULL, **Z = NULL, *a = NULL, *z = NULL; // @heap_allocated
	arrpush(A, a = x);
	for (int l = 1; l < L; ++l) {
		// calculate weighted sum: matrix_vector_mul, vector_add
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
	// δL = hadamard(y - aL, sigma_prime(zL))
	float *deltas = NULL;
	for (int j = 0; j < net->layers[L-1]; ++j) {
		float delta = (y[j] - A[L-1][j]) * sigmoid_prime(Z[L-1][j]);
		for (int k = 0; k < net->layers[L-2]; ++k) {
			nabla_w[L-1][j][k] += delta * A[L-2][k];
		}
		nabla_b[L-1][j] += delta;
		arrpush(deltas, delta);
	}
	// for l < L: δ(l) = hadamard(dot(W(l+1), δ(l+1)), sigma_prime(Z(l)))
	for (int l = L-2; l > 1; --l) {
		float *new_deltas = NULL;
		for (int j = 0; j < net->layers[l+1]; ++j) {
			float delta = 0.0f;
			for (int k = 0; k < net->layers[l]; ++k) {
				delta += deltas[k] * net->weights[l+1][j][k];
			}
			delta *= sigmoid_prime(Z[l][j]);
			// update weights and biases gradients
			for (int k = 0; k < net->layers[l]; ++k) {
				nabla_w[l][j][k] += delta * A[l][k];
			}
			nabla_b[L-1][j] += delta;
			arrpush(new_deltas, delta);
		}
		arrfree(deltas);
		deltas = new_deltas;
	}
	arrfree(deltas);
	// narrfree(A, 2);
	// narrfree(Z, 2);
}

// mini_batch = [
// 	batch: [x, y, x', y' ...],
// 	batch: [x, y, x', y' ...],
// ]
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
	for (int b = 0; b < arrlen(mini_batch_set); b += 2) {
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
SGD(Network *net, SetEntry *training_data, size_t epochs, size_t mini_batch_size, float eta) {
	assert(arrlen(training_data) == net->layers[0] && "training_data's length must match input layer's");
	for (int e = 0; e < epochs; ++e) {
		shuffle_set(training_data);
		SetEntry **mini_batches = generate_mini_batches(training_data, mini_batch_size);
		for (int b = 0; b < arrlen(mini_batches); ++b) {
			update_mini_batch(net, mini_batches[b], eta);
			arrfree(mini_batches[b]);
		}
		arrfree(mini_batches);
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

	// InitWindow(WIDTH, HEIGHT, "Hello World!");
	// SetTargetFPS(1);
    	// while (!WindowShouldClose()) {
    	//     	BeginDrawing();
	// 		ClearBackground(BLACK);
	// 		DrawNeurons(net);
    	//     	EndDrawing();
    	// }
    	// CloseWindow();

    	return 0;
}
