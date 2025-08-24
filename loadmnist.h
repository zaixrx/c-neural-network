/*
source in case I wanted to write an http client to fetch this:
wget https://systemds.apache.org/assets/datasets/mnist/train-images-idx3-ubyte.gz
wget https://systemds.apache.org/assets/datasets/mnist/train-labels-idx1-ubyte.gz
wget https://systemds.apache.org/assets/datasets/mnist/t10k-images-idx3-ubyte.gz
wget https://systemds.apache.org/assets/datasets/mnist/t10k-labels-idx1-ubyte.gz
*/

#ifndef LOAD_MNIST_H
#define LOAD_MNIST_H

#include <stdio.h>
#include <stdint.h>
#define STB_DS_IMPLEMENTATION
#include "include/stb_ds.h"

#define PATH_TRAIN_IMAGE "./dataset/train-images-idx3-ubyte"
#define PATH_TRAIN_LABEL "./dataset/train-labels-idx1-ubyte"

typedef struct {
	uint8_t *x;
	uint8_t y;
} SetEntry;

SetEntry *get_training_set(size_t set_size);

#endif

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

void parse_digits(parser_t *p, SetEntry *set, size_t set_size) {
	PARSE_HEADER(p);
	assert(dims_c == 3 && dims[0] == set_size && dims[1] == 28 && dims[2] == 28);
	size_t res = dims[1] * dims[2];
	for (int e = 0; e < dims[0]; ++e) {
		set[e].x = malloc(res);
		memcpy(set[e].x, p->buf + p->off, res);
		p->off += res;
	}
}

void parse_labels(parser_t *p, SetEntry *set, size_t set_size) {
	PARSE_HEADER(p);
	assert(dims_c == 1 && dims[0] == set_size);
	for (int e = 0; e < dims[0]; ++e) {
		set[e].y = parser_u8(p);
	}
}

// @allocated
void *read_file(char *file_path, size_t *size) {
	FILE *stream = fopen(file_path, "rb");
	assert(stream != NULL);
	assert(fseek(stream, 0, SEEK_END) != -1);
	assert((*size = ftell(stream)) != -1);
	assert(fseek(stream, 0, SEEK_SET) != -1);
	void *buf = malloc(*size);
	assert(fread(buf, 1, *size, stream) != -1);
	fclose(stream);
	return buf;
}

void reset_parser(parser_t p) {
	free(p.buf);
	p.buf = NULL;
	p.size = p.off = 0;
}

SetEntry *get_training_set(size_t set_size) {
	size_t file_size;
	uint8_t *buf;
	parser_t parser;
	SetEntry *entries;
	arrsetlen(entries, set_size);

	buf = read_file(PATH_TRAIN_IMAGE, &file_size);
	parser.buf = buf; parser.size = file_size;
	parse_digits(&parser, entries, set_size);
	reset_parser(parser);

	buf = read_file(PATH_TRAIN_LABEL, &file_size);
	parser.buf = buf; parser.size = file_size;
	parse_labels(&parser, entries, set_size);
	reset_parser(parser);

	return entries;
}
#endif
