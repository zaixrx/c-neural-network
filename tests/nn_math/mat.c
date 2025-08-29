#define STB_DS_IMPLEMENTATION
#define NN_MATH_IMPLEMENTATION
#include "nn_math.h"

void test_create_destroy() {
    mat_t m = mat_new(3, 3);
    assert(m != NULL);
    mat_destroy(m);
}

void test_scale_identity() {
    mat_t m = mat_new(2, 2);
    m[0][0] = 1; m[0][1] = 2;
    m[1][0] = 3; m[1][1] = 4;

    mat_scale(m, 2);

    assert(m[0][0] == 2);
    assert(m[0][1] == 4);
    assert(m[1][0] == 6);
    assert(m[1][1] == 8);

    mat_destroy(m);
}

void test_addition() {
    mat_t a = mat_new(2, 2);
    mat_t b = mat_new(2, 2);

    a[0][0] = 1; a[0][1] = 2;
    a[1][0] = 3; a[1][1] = 4;

    b[0][0] = 5; b[0][1] = 6;
    b[1][0] = 7; b[1][1] = 8;

    mat_operate(a, 2, (MatOp){ LOAD, a }, (MatOp){ ADD, b });

    assert(a[0][0] == 6);
    assert(a[0][1] == 8);
    assert(a[1][0] == 10);
    assert(a[1][1] == 12);

    mat_destroy(a);
    mat_destroy(b);
}

void test_subtraction() {
    mat_t a = mat_new(2, 2);
    mat_t b = mat_new(2, 2);

    a[0][0] = 5; a[0][1] = 7;
    a[1][0] = 9; a[1][1] = 11;

    b[0][0] = 1; b[0][1] = 2;
    b[1][0] = 3; b[1][1] = 4;

    mat_operate(a, 2, (MatOp){ LOAD, a }, (MatOp){ SUB, b });

    assert(a[0][0] == 4);
    assert(a[0][1] == 5);
    assert(a[1][0] == 6);
    assert(a[1][1] == 7);

    mat_destroy(a);
    mat_destroy(b);
}

void test_multiplication() {
    mat_t a = mat_new(2, 2);
    mat_t b = mat_new(2, 2);

    a[0][0] = 2; a[0][1] = 3;
    a[1][0] = 4; a[1][1] = 5;

    b[0][0] = 6; b[0][1] = 7;
    b[1][0] = 8; b[1][1] = 9;

    mat_operate(a, 2, (MatOp){ LOAD, a }, (MatOp){ MUL, b });

    assert(a[0][0] == 12); assert(a[0][1] == 21);
    assert(a[1][0] == 32); assert(a[1][1] == 45);

    mat_destroy(a);
    mat_destroy(b);
}

void test_division() {
    mat_t a = mat_new(2, 2);
    mat_t b = mat_new(2, 2);

    a[0][0] = 10; a[0][1] = 20;
    a[1][0] = 30; a[1][1] = 40;

    b[0][0] = 2; b[0][1] = 4;
    b[1][0] = 5; b[1][1] = 8;

    mat_operate(a, 2, (MatOp){ LOAD, a }, (MatOp){ DIV, b });

    assert(a[0][0] == 5);
    assert(a[0][1] == 5);
    assert(a[1][0] == 6);
    assert(a[1][1] == 5);

    mat_destroy(a);
    mat_destroy(b);
}

void test_chained_operations() {
    mat_t a = mat_new(2, 2);
    mat_t b = mat_new(2, 2);
    mat_t c = mat_new(2, 2);

    a[0][0] = 1; a[0][1] = 2;
    a[1][0] = 3; a[1][1] = 4;

    b[0][0] = 5; b[0][1] = 6;
    b[1][0] = 7; b[1][1] = 8;

    c[0][0] = 2; c[0][1] = 2;
    c[1][0] = 2; c[1][1] = 2;

    mat_operate(a, 4, (MatOp){ LOAD, a }, (MatOp){ ADD, b }, (MatOp){ MUL, c }, (MatOp){ SUB, c });

    // (a+b)*c - c
    assert(a[0][0] == ((1+5)*2 - 2)); // 10
    assert(a[0][1] == ((2+6)*2 - 2)); // 14
    assert(a[1][0] == ((3+7)*2 - 2)); // 18
    assert(a[1][1] == ((4+8)*2 - 2)); // 22

    mat_destroy(a);
    mat_destroy(b);
    mat_destroy(c);
}

int main() {
    test_create_destroy();
    test_scale_identity();
    test_addition();
    test_subtraction();
    test_multiplication();
    test_division();
    test_chained_operations();

    printf("All matrix tests passed!\n");
    return 0;
}
