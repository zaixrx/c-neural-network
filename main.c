#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include "include/raylib.h"
#define STB_DS_IMPLEMENTATION
#include "include/stb_ds.h"

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
		printf("%f\n", x);
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

float
sigmoid(float x) {
	return 1 / (1 + exp(-x));
}

float
*feed_forward(Network *net, float *data) {
	assert(arrlen(data) == net->layers[0] && "input's length must match input layer's");
	for (int l = 1; l < arrlen(net->layers); ++l) {
		float *output = NULL;
		for (int n = 0; n < net->layers[l]; ++n) {
			float sum = net->biases[l][n]; // b
			for (int pn = 0; pn < net->layers[l-1]; ++pn) {
				sum += data[pn] * net->weights[l-1][n][pn]; // Σ (Wjk * Ak) where k(L-1) connects j(L)
			}
			arrpush(output, sum);
		}
		arrfree(data);
		data = output;
	}
	return data;
}

int rnd() {
	srand(time(NULL));
	return rand();
}

void
shuffle_array(float *arr) {
	for (int i = 0; i < arrlen(arr); ++i) {
		int j = rnd() % arrlen(arr);
		arr[i] += arr[j];
		arr[j] -= arr[i];
		arr[i] -= arr[j];
	}
}

float
**generate_mini_batches(float *data, size_t mini_batch_size) {
	float **mini_batches = NULL;
	for (int i = 0; i < arrlen(data); i += mini_batch_size) {
		float *mini_batch = NULL;
		arrsetlen(mini_batch, mini_batch_size);
		memcpy(mini_batch, data + i, mini_batch_size);
		arrpush(mini_batches, mini_batch);
	}
	return NULL;
}

void
backprop(float *x, float *y) {
	abort();
}

// mini_batch = [[x, y], [x', y'], ...]
void
update_mini_batch(Network *net, float *mini_batch, float eta) {
	float **nabla_b = NULL;
	float ***nabla_w = NULL;
	assert(arrlen(mini_batch) % 2 == 0);
	for (int b = 0; b < arrlen(mini_batch); b += 2) {
		float x = mini_batch[b], y = mini_batch[b + 1];
		backprop(x, y);
	}
	for (int bl = 0; bl < arrlen(net->biases); ++bl) {
		for (int b = 0; b < arrlen(net->biases[bl]); ++b) {
			net->biases[bl][b] += -eta * 0.5f / arrlen(mini_batch) * nabla_b[bl][b];
		}
	}
	for (int wl= 0; wl < arrlen(net->weights); ++wl) {
		for (int w = 0; w < arrlen(net->weights[wl]); ++w) {
			for (int pw = 0; pw < arrlen(net->weights[pw][w]); ++pw) {
				net->weights[wl][w][pw] += -eta * 0.5f / arrlen(mini_batch) * nabla_w[wl][w][pw];
			}
		}
	}
}

void
SGD(Network *net, float *training_data, size_t epochs, size_t mini_batch_size, float eta) {
	assert(arrlen(training_data) == net->layers[0] && "training_data's length must match input layer's");
	for (int e = 0; e < epochs; ++e) {
		shuffle_array(training_data);
		float **mini_batches = generate_mini_batches(training_data, mini_batch_size);
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
	{
		size_t *layers = NULL;
		arrpush(layers, 5);
		arrpush(layers, 7);
		arrpush(layers, 3);
		arrpush(layers, 8);
		arrpush(layers, 1);
		net = new_network(layers);
	}

	InitWindow(WIDTH, HEIGHT, "Hello World!");
	SetTargetFPS(1);

    	while (!WindowShouldClose()) {
    	    	BeginDrawing();
			ClearBackground(BLACK);
			DrawNeurons(net);
    	    	EndDrawing();
    	}

    	CloseWindow();

    	return 0;
}
