#include "nn.h"
#include "nn_math.h"
#include "stb_ds.h"
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <stdio.h>
#include <stdbool.h>

#define TODO(message) do { fprintf(stderr, "%s:%d: TODO: %s\n", __FILE__, __LINE__, message); abort(); } while(0)

typedef enum {
	AFN_RELU,
	AFN_SIGMOID,
} AFN_TYPE;

static double d_sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

static void sigmoid(vec_t dst, vec_t src) {
	assert(arrlen(src) == arrlen(dst));
	for (int i = 0; i < arrlen(src); ++i) {
		dst[i] = 1.0 / (1.0 + exp(-src[i]));
	}
}

static double d_sigmoid_prime(double x) {
	double e = exp(-x);
	return e / ((1 + e) * (1 + e));
}

static void sigmoid_prime(vec_t dst, vec_t src) {
	assert(arrlen(src) == arrlen(dst));
	for (int i = 0; i < arrlen(src); ++i) {
		double e = exp(-src[i]);
		dst[i] = e / ((1 + e) * (1 + e));
	}
}

static inline double rand_uniform() {
    return (rand() + 1.0) / (RAND_MAX + 2.0); // avoid 0
}

// standard normal via Box-Muller
double randn() {
    static bool has_spare = 0;
    static double spare;

    if (has_spare) return has_spare = false, spare;

    double u, v;
    u = rand_uniform();
    v = rand_uniform();

    double mag = sqrt(-2.0 * log(u));
    has_spare = true, spare = mag * sin(2.0 * M_PI * v);
    return mag * cos(2.0 * M_PI * v);
}

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

void network_create(Network *net, size_t *sizes) {
	net->sizes = sizes;
	net->weights = new_mat_arr(sizes);
	net->biases = new_vec_arr(sizes);
	srand(time(NULL));
	for (size_t l = 0; l < arrlen(net->weights); ++l) {
		for (size_t i = 0; i < arrlen(net->weights[l]); ++i) {
			net->biases[l][i] = randn();
			for (size_t j = 0; j < arrlen(net->weights[l][i]); ++j) {
				net->weights[l][i][j] = randn();
			}
		}
	}
}

void network_destroy(Network *net) {
	free_mat_arr(net->weights);
	free_vec_arr(net->biases);
}

static void shuffle_set(DataEntry *set) {
	DataEntry temp;
	for (int i = 0; i < arrlen(set); ++i) {
		int r = rand() % arrlen(set);
		temp = set[i];
		set[i] = set[r];
		set[r] = temp;
	}
}

static DataEntry **get_batches_from_set(DataEntry *set, size_t batch_size) {
	DataEntry **batches = NULL;
	for (size_t b = 0; b + batch_size <= arrlen(set); b += batch_size) {
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
		printf("DEBUG :: analysing %zu batches\n", arrlen(batches));
		clock_t start = clock();
		for (size_t b = 0; b < arrlen(batches); ++b) {
			network_update_batch(net, batches[b], lrate);
			// free(batch);
		}
		printf("DEBUG :: took: %lfs\n", (double)(clock()-start)/CLOCKS_PER_SEC);
		arrfree(batches);
		if (test_set) {
			size_t t, ts; // test, test_success
			for (t = 0, ts = 0; t < arrlen(test_set); ++t) {
				ts += network_test(net, test_set[t]);
			}
			printf("INFO :: Epoch %zu: %zu/%zu\n", e, ts, t);
			continue;
		}
		printf("INFO :: Epoch %zu\n", e);
	}
}

int network_test(Network *net, DataEntry entry) {
	vec_t y = entry.x;
	vec_t *Y = new_vec_arr(net->sizes);
	for (size_t l = 0; l < arrlen(Y); ++l) {
		mat_vec_dot(Y[l], net->weights[l], y);
		vec_operate(Y[l], 1, (VecOp){ ADD, net->biases[l] });
		sigmoid(Y[l], Y[l]);
		y = Y[l];
	}
	size_t max = 0;
	for (size_t i = 1; i < arrlen(y); ++i) if (y[i] > y[max]) max = i;
	int ret = entry.y[max] >= 1;
	free_vec_arr(Y);
	return ret;
}

void network_update_batch(Network *net, DataEntry *batch, double lrate) {
	// allocate gradients
	mat_t *grad_weights = new_mat_arr(net->sizes);
	vec_t *grad_biases = new_vec_arr(net->sizes);

	// calculate gradients
	for (size_t b = 0; b < arrlen(batch); ++b) {
		network_backprop(net, batch[b], grad_weights, grad_biases);
	}
	
	double scaler = -(lrate / arrlen(batch));
	for (size_t l = 0; l < arrlen(net->sizes)-1; ++l) {
		mat_scale(grad_weights[l], scaler);
		vec_scale(grad_biases[l], scaler);
		mat_operate(net->weights[l], 1, (MatOp){ ADD, grad_weights[l] });
		vec_operate(net->biases[l], 1, (VecOp){ ADD, grad_biases[l] });
	}

	// free gradients
	free_mat_arr(grad_weights);
	free_vec_arr(grad_biases);
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
	for (ssize_t l = arrlen(D) - 1; l >= 0; --l) {
		// update gradients
		vec_t v = l > 0 ? A[l-1] : entry.x;
		for (size_t i = 0; i < arrlen(D[l]); ++i) {
			if (l == arrlen(D) - 1) {
				D[l][i] = A[l][i] - entry.y[i];
			} else {
				double val = 0;
				for (size_t j = 0; j < arrlen(net->weights[l+1]); ++j) {
					val += net->weights[l+1][j][i] * D[l+1][j];
				}
				D[l][i] = val;
			}
			D[l][i] *= d_sigmoid_prime(Z[l][i]);
			grad_biases[l][i] += D[l][i];
			for (size_t j = 0; j < arrlen(v); ++j) {
				grad_weights[l][i][j] += v[j] * D[l][i];
			}
		}
	}
	free_vec_arr(Z);
	free_vec_arr(A);
	free_vec_arr(D);
}

static struct {
	char *buf;
	size_t size;
	size_t offset;
} Packet = {0};

void packet_init(char *buf, size_t size) {
	Packet.buf = buf;
	Packet.size = size;
	Packet.offset = 0;
}

void packet_write_u64(uint64_t val) {
	assert(Packet.offset < Packet.size && "buffer overflow: packet_write_u64");
	((uint64_t*)(Packet.buf + Packet.offset))[0] = val;
	Packet.offset += sizeof(uint64_t);
}

void packet_write_f64(double val) {
	assert(Packet.offset < Packet.size && "buffer overflow: packet_write_f64");
	((double*)(Packet.buf + Packet.offset))[0] = val;
	Packet.offset += sizeof(double);
}

uint64_t packet_read_u64() {
	assert(Packet.offset < Packet.size && "buffer overflow: packet_read_u64");
	uint64_t val = ((uint64_t*)(Packet.buf + Packet.offset))[0];
	Packet.offset += sizeof(uint64_t);
	return val;
}

double packet_read_f64() {
	assert(Packet.offset < Packet.size && "buffer overflow: packet_read_f64");
	double val = ((double*)(Packet.buf + Packet.offset))[0];
	Packet.offset += sizeof(double);
	return val;
}

NN_CODE network_export(Network *net, const char *file_path) {
	size_t L = arrlen(net->sizes);
	size_t size = (L + 1) * sizeof(size_t); // header
	for (size_t l = 1; l < L; ++l) {
		size += sizeof(double) * net->sizes[l] * (1 + net->sizes[l-1]); // payload
	}
	char *buf = malloc(size);
	if (!buf) return NN_CODE_FAILURE;

	packet_init(buf, size);

	// print_header: u64
	// 	sizes_length-1, ...sizes[1:]
	packet_write_u64(L);
	for (size_t i = 0; i < L; ++i) {
		packet_write_u64(net->sizes[i]);
	}
	
	// print body: f64
	// 	...weights
	// 	...biases
	for (size_t l = 0; l < L-1; ++l) {
		for (size_t i = 0; i < arrlen(net->weights[l]); ++i) {
			for (size_t j = 0; j < arrlen(net->weights[l][i]); ++j) {
				packet_write_f64(net->weights[l][i][j]);
			}
		}
		for (size_t i = 0; i < arrlen(net->biases[l]); ++i) {
			packet_write_f64(net->biases[l][i]);
		}
	}

	FILE* stream = fopen(file_path, "w");
	if (!stream) return NN_CODE_FAILURE;
	size_t w_size = fwrite(buf, size, 1, stream);
	if (w_size != size) return NN_CODE_FAILURE;
	if (fclose(stream) == EOF) return NN_CODE_FAILURE;

	return NN_CODE_SUCCESS;
}

NN_CODE network_import(Network *net, const char *file_path) {
	FILE *stream;
	char *buf;
	size_t size;

	stream = fopen(file_path, "r");

	if (!stream) return NN_CODE_FAILURE;
	if (fseek(stream, 0, SEEK_END) == -1) return NN_CODE_FAILURE;
	if ((size = ftell(stream)) == -1) return NN_CODE_FAILURE;
	if (fseek(stream, 0, SEEK_SET) == -1) return NN_CODE_FAILURE;
	assert((buf = malloc(size)));
	if (fread(buf, 1, size, stream) != size) return NN_CODE_FAILURE;
	if (fclose(stream) == EOF) return NN_CODE_FAILURE;

	packet_init(buf, size);

	// print_header: u64
	// 	sizes_length, ...sizes
	size_t L = packet_read_u64();
	size_t *sizes = NULL;
	arrsetlen(sizes, L);
	for (size_t i = 0; i < L; ++i) {
		sizes[i] = packet_read_u64();
	}
	
	printf("Header { Layers: %zu }\n", L);

	// print body: f64
	// 	...weights
	// 	...biases
	mat_t *weights = NULL;
	vec_t *biases  = NULL;
	arrsetlen(weights, L-1);
	arrsetlen(biases , L-1);
	for (size_t l = 0; l < L-1; ++l) {
		weights[l] = mat_new(sizes[l+1], sizes[l]);
		for (size_t i = 0; i < arrlen(weights[l]); ++i) {
			for (size_t j = 0; j < arrlen(weights[l][i]); ++j) {
				weights[l][i][j] = packet_read_f64();
			}
		}
		biases[l] = vec_new(sizes[l+1]);
		for (size_t i = 0; i < arrlen(biases[l]); ++i) {
			biases[l][i] = packet_read_f64();
		}
		printf("Payload[%zu] { mat(%zu x %zu), vec(%zu) }\n", l, sizes[l+1], sizes[l], sizes[l+1]);
	}

	net->sizes = sizes;
	net->weights = weights;
	net->biases = biases;

	return NN_CODE_SUCCESS;
}
