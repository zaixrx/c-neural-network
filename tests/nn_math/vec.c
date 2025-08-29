#include <assert.h>
#include <stdio.h>
#define STB_DS_IMPLEMENTATION
#define NN_MATH_IMPLEMENTATION
#include "nn_math.h"

void test_vector_creation() {
    vec_t v = vec_new(0);
    assert(v == NULL);
    assert(arrlen(v) == 0);

    vec_t u = vec_new(5);
    assert(arrlen(u) == 5);
    for (size_t i = 0; i < arrlen(u); ++i)
        assert(u[i] == 0.0);

    vec_destroy(v);
    vec_destroy(u);
    printf("creation tests passed\n");
}

void test_vector_scaling() {
    vec_t v = vec_new(4);
    for (size_t i = 0; i < 4; ++i) v[i] = i+1; // 1,2,3,4
    vec_scale(v, 2.0); // → 2,4,6,8
    assert(v[0] == 2.0 && v[1] == 4.0 && v[2] == 6.0 && v[3] == 8.0);
    vec_scale(v, 0.0); // → 0,0,0,0
    for (size_t i = 0; i < 4; ++i) assert(v[i] == 0.0);
    vec_destroy(v);

    vec_t single = vec_new(1);
    single[0] = 42.0;
    vec_scale(single, -1.0);
    assert(single[0] == -42.0);
    vec_destroy(single);

    printf("scaling tests passed\n");
}

void test_vector_operations() {
    vec_t a = vec_new(3);
    vec_t b = vec_new(3);
    for (size_t i = 0; i < 3; ++i) { a[i] = i+1; b[i] = (i+1)*10; } // a = 1,2,3; b = 10,20,30

    vec_operate(a, 2,
        (VecOp){ LOAD, a },
        (VecOp){ ADD, b }
    ); // expect 11,22,33
    assert(a[0] == 11 && a[1] == 22 && a[2] == 33);

    vec_operate(a, 2,
        (VecOp){ LOAD, a },
        (VecOp){ SUB, b }
    ); // expect 1,2,3 again
    assert(a[0] == 1 && a[1] == 2 && a[2] == 3);

    vec_operate(a, 2,
        (VecOp){ LOAD, a },
        (VecOp){ MUL, b }
    ); // expect 10,40,90
    assert(a[0] == 10 && a[1] == 40 && a[2] == 90);

    vec_operate(a, 2,
        (VecOp){ LOAD, a },
        (VecOp){ DIV, b }
    ); // expect 1,2,3 again
    assert(a[0] == 1 && a[1] == 2 && a[2] == 3);

    vec_destroy(a);
    vec_destroy(b);
    printf("operations tests passed\n");
}

void test_vector_integration() {
    vec_t a = vec_new(10);
    vec_t b = vec_new(10);
    vec_t c = vec_new(10);
    vec_t d = vec_new(10);
    for (size_t i = 0; i < 10; ++i) {
        a[i] = i + 1;
        b[i] = 10 - i;
        c[i] = -i;
        d[i] = -10 + i;
    }
    vec_scale(a, 10);
    vec_operate(a, 5,
        (VecOp){ LOAD, a },
        (VecOp){ ADD, b },
        (VecOp){ SUB, c },
        (VecOp){ MUL, d }
    );
    for (size_t i = 1; i < arrlen(a); ++i) {
	double expected = (10*(i+1) + b[i] - c[i]) * d[i];
	assert(a[i] == expected);
    }
    vec_destroy(a);
    vec_destroy(b);
    vec_destroy(c);
    vec_destroy(d);
    printf("integration test passed\n");
}

int main() {
    test_vector_creation();
    test_vector_scaling();
    test_vector_operations();
    test_vector_integration();
    printf("ALL VECTOR TESTS PASSED\n");
    return 0;
}
