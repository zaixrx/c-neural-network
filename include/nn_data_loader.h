/*
source in case I wanted to write an http client to fetch this:
wget https://systemds.apache.org/assets/datasets/mnist/train-images-idx3-ubyte.gz
wget https://systemds.apache.org/assets/datasets/mnist/train-labels-idx1-ubyte.gz
wget https://systemds.apache.org/assets/datasets/mnist/t10k-images-idx3-ubyte.gz
wget https://systemds.apache.org/assets/datasets/mnist/t10k-labels-idx1-ubyte.gz
*/

#ifndef LOAD_MNIST_H
#define LOAD_MNIST_H

#include <stdint.h>
#include <stdio.h>
#include "stb_ds.h"
#include "assert.h"

#define PREFIX "data"
#define TRAIN_SET_IMAGE PREFIX "/train-images-idx3-ubyte"
#define TRAIN_SET_LABEL PREFIX "/train-labels-idx1-ubyte"
#define TEST_SET_IMAGE PREFIX "/t10k-images-idx3-ubyte"
#define TEST_SET_LABEL PREFIX "/t10k-labels-idx1-ubyte"
#define TRAIN_SET_SIZE 6e4
#define TEST_SET_SIZE 1e4
#define Y_SIZE 10

typedef struct {
	double *x;
	double *y;
} DataEntry;

DataEntry *load_training_set();
DataEntry *load_test_set();

#endif // LOAD_MNIST_H

#ifdef LOAD_MNIST_IMPLEMENTAION

typedef struct {
	uint8_t *buf;
	size_t size;
	size_t off;
} parser_t;

int parser_eof(parser_t *p, size_t size) {
	return p->off + size > p->size;
}

uint8_t parser_u8(parser_t *p) {
	assert(!parser_eof(p, 1));
	return p->buf[p->off++];
}

uint32_t parser_u32(parser_t *p) {
	assert(!parser_eof(p, 4));
	return p->off += 4, *(uint32_t*)(p->buf + p->off - 4);
}

#define PARSE_HEADER(p) \
	p->off = 2; \
	assert(parser_u8(p) == 0x08); \
	uint8_t dims_c = parser_u8(p); \
	uint32_t dims[dims_c]; \
	for (uint8_t d = 0; d < dims_c; ++d) { \
		dims[d] = parser_u32(p); \
		dims[d] = ((dims[d] >> 24)& 0xFF) | \
                          ((dims[d] << 8) & 0xFF0000) | \
                          ((dims[d] >> 8) & 0xFF00) | \
                          ((dims[d] << 24)& 0xFF000000); \
	}

void parse_digits(parser_t *p, DataEntry *set, size_t set_size) {
	PARSE_HEADER(p);
	assert(dims_c == 3 && dims[0] == set_size && dims[1] == 28 && dims[2] == 28);
	size_t res = dims[1] * dims[2];
	for (int e = 0; e < dims[0]; ++e) {
		set[e].x = NULL;
		arrsetlen(set[e].x, res);
		for (int i = 0; i < res; ++i) {
			set[e].x[i] = (double)parser_u8(p) / 255.0;
		}
	}
}

void parse_labels(parser_t *p, DataEntry *set, size_t set_size) {
	PARSE_HEADER(p);
	assert(dims_c == 1 && dims[0] == set_size);
	for (int e = 0; e < dims[0]; ++e) {
		uint8_t digit = parser_u8(p);
		set[e].y = NULL;
		arrsetlen(set[e].y, Y_SIZE);
		for (uint8_t i = 0; i < Y_SIZE; ++i) {
			set[e].y[i] = (double)((i + 1) == digit);
		}
	}
}

// @allocated
uint8_t *read_file(const char *file_path, size_t *size) {
	FILE *stream = fopen(file_path, "rb");
	assert(stream != NULL);
	assert(fseek(stream, 0, SEEK_END) != -1);
	assert((*size = ftell(stream)) != -1);
	assert(fseek(stream, 0, SEEK_SET) != -1);
	uint8_t *buf = (uint8_t*)malloc(*size);
	assert(fread(buf, 1, *size, stream) != -1);
	fclose(stream);
	return buf;
}

void reset_parser(parser_t p) {
	free(p.buf);
	p.buf = NULL;
	p.size = p.off = 0;
}

DataEntry *load_set(const char *image_path, const char *label_path, size_t set_size) {
	size_t file_size;
	uint8_t *buf;
	parser_t parser;
	DataEntry *entries = NULL;
	arrsetlen(entries, set_size);

	buf = read_file(image_path, &file_size);
	parser.buf = buf; parser.size = file_size;
	parse_digits(&parser, entries, set_size);
	reset_parser(parser);

	buf = read_file(label_path, &file_size);
	parser.buf = buf; parser.size = file_size;
	parse_labels(&parser, entries, set_size);
	reset_parser(parser);

	return entries;
}

DataEntry *load_training_set() {
	return load_set(TRAIN_SET_IMAGE, TRAIN_SET_LABEL, TRAIN_SET_SIZE);
}

DataEntry *load_test_set() {
	return load_set(TEST_SET_IMAGE, TEST_SET_LABEL, TEST_SET_SIZE);
}
#endif
