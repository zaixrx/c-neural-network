#ifndef NN_MATH_H
#define NN_MATH_H

#include <assert.h>
#include <stddef.h>
#include <stdarg.h>
#include <stdio.h>
#include "stb_ds.h"

typedef double* vec_t;
typedef vec_t* mat_t;

typedef enum { LOAD, ADD, SUB, MUL, DIV } OpType;

typedef struct {
	OpType type;
	vec_t val;
} VecOp;

typedef struct {
	OpType type;
	mat_t val;
} MatOp;

vec_t vec_new(size_t);
void vec_destroy(vec_t);
void vec_operate(vec_t, size_t, ...);
void vec_scale(vec_t, double);
void vec_print(vec_t);
void vec_print_dims(vec_t);

void vecT_vec_dot(mat_t, vec_t, vec_t);
void mat_vec_dot(vec_t, mat_t, vec_t);
void matT_vec_dot(vec_t, mat_t, vec_t);

mat_t mat_new(size_t, size_t);
void mat_destroy(mat_t);
void mat_operate(mat_t, size_t, ...);
void mat_scale(mat_t, double);
void mat_print(mat_t);
void met_print_dims(mat_t);

#endif //NN_MATH_H

#ifdef NN_MATH_IMPLEMENTATION

static inline double operate(double x, double y, OpType o) {
	switch (o) {
		case LOAD: return y;
		case ADD: return x + y;
		case SUB: return x - y;
		case MUL: return x * y;
		case DIV: return x / y;
	}
}

vec_t vec_new(size_t len) {
	vec_t vec = NULL;
	for (size_t r = 0; r < len; ++r) {
		arrpush(vec, 0);
	}
	return vec;
}

void vec_destroy(vec_t v) {
	arrfree(v);
}

void vec_operate(vec_t src, size_t n, ...) {
	va_list args;
	va_start(args, n);
	for (size_t o = 0; o < n; ++o) {
		VecOp op = va_arg(args, VecOp);
		assert(arrlen(src) == arrlen(op.val) && "vec_operate");
		for (size_t i = 0; i < arrlen(src); ++i) {
			src[i] = operate(src[i], op.val[i], op.type);
		}
	}
	va_end(args);
}

void vec_scale(vec_t src, double scaler) {
	for (size_t i = 0; i < arrlen(src); ++i) {
		src[i] *= scaler;
	}
}

void vec_print(vec_t v) {
	printf("[ ");
	for (size_t i = 0; i < arrlen(v); ++i) {
		printf("%lf", v[i]);
	}
	printf(" ]\n");
}

void vec_print_dims(vec_t vec) {
	printf("arr.len(): %zu\n", arrlen(vec));
}

mat_t mat_new(size_t row, size_t col) {
	mat_t mat = NULL;
	for (size_t r = 0; r < row; ++r) {
		arrpush(mat, vec_new(col));
	}
	return mat;
}

void mat_destroy(mat_t m) {
	for (size_t i = 0; i < arrlen(m); ++i) {
		vec_destroy(m[i]);
	}
	arrfree(m);
}

void mat_operate(mat_t src, size_t n, ...) {
	va_list args;
	va_start(args, n);
	for (size_t o = 0; o < n; ++o) {
		MatOp op = va_arg(args, MatOp);
		assert(arrlen(src) == arrlen(op.val) && "vec_operate");
		for (size_t i = 0; i < arrlen(src); ++i) {
			assert(arrlen(src[i]) == arrlen(op.val[i]) && "vec_operate");
			for (size_t j = 0; j < arrlen(src[i]); ++j) {
				src[i][j] = operate(src[i][j], op.val[i][j], op.type);
			}
		}
	}
	va_end(args);
}

void mat_scale(mat_t mat, double scaler) {
	for (size_t i = 0; i < arrlen(mat); ++i) {
		for (size_t j = 0; j < arrlen(mat[i]); ++j) {
			mat[i][j] *= scaler;
		}
	}
}

void mat_print(mat_t m) {
	printf("[\n");
	for (size_t i = 0; i < arrlen(m); ++i) {
		vec_print(m[i]);
	}
	printf("]\n");
}

void vecT_vec_dot(mat_t dst, vec_t a, vec_t b) {
	assert(arrlen(dst) == arrlen(b));
	for (size_t i = 0; i < arrlen(b); ++i) {
		assert(arrlen(dst[i]) == arrlen(a));
		for (size_t j = 0; j < arrlen(a); ++j) {
			dst[i][j] = a[j] * b[i];
		}
	}
}

void mat_vec_dot(vec_t dst, mat_t mat, vec_t vec) {
	assert(arrlen(mat) == arrlen(dst) && "mat_vec_dot");
	assert(arrlen(mat[0]) == arrlen(vec) && "mat_vec_dot");
	for (size_t i = 0; i < arrlen(mat); ++i) {
		double val = 0;
		for (size_t j = 0; j < arrlen(mat[i]); ++j) {
			val += mat[i][j] * vec[j];
		}
		dst[i] = val;
	}
}

void matT_vec_dot(vec_t dst, mat_t mat, vec_t vec) {
	assert(arrlen(mat[0]) == arrlen(dst) && "matT_vec_dot");
	assert(arrlen(mat) == arrlen(vec) && "matT_vec_dot");
	for (size_t j = 0; j < arrlen(mat[0]); ++j) {
		double val = 0;
		for (size_t i = 0; i < arrlen(mat); ++i) {
			val += mat[i][j] * vec[i];
		}
		dst[j] = val;
	}
}

void mat_print_dims(mat_t mat) {
	printf("mat.len(): %zu\n[\n", arrlen(mat));
	for (size_t i = 0; i < arrlen(mat); ++i) {
		vec_print_dims(mat[i]);
	}
	printf("]\n");
}

#endif
