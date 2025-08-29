#define STB_DS_IMPLEMENTATION
#define NN_MATH_IMPLEMENTATION
#include "nn_math.h"

void test_vecT_vec_dot() {
    vec_t u = vec_new(3);
    vec_t v = vec_new(2);
    mat_t M = mat_new(2, 3);

    // u = [1, 2], v = [3, 4, 5]
    u[0] = 3; u[1] = 4; u[2] = 5;
    v[0] = 1; v[1] = 2;

    vecT_vec_dot(M, u, v);

    // Expected M = u * v^T =
    // [1*3, 1*4, 1*5]
    // [2*3, 2*4, 2*5]
    assert(M[0][0] == 3 && M[0][1] == 4 && M[0][2] == 5);
    assert(M[1][0] == 6 && M[1][1] == 8 && M[1][2] == 10);

    vec_destroy(u);
    vec_destroy(v);
    mat_destroy(M);
}

void test_mat_vec_dot() {
    mat_t M = mat_new(2, 3);
    vec_t v = vec_new(3);
    vec_t out = vec_new(2);

    // M = [[1,2,3],[4,5,6]]
    M[0][0] = 1; M[0][1] = 2; M[0][2] = 3;
    M[1][0] = 4; M[1][1] = 5; M[1][2] = 6;
    // v = [7,8,9]
    v[0] = 7; v[1] = 8; v[2] = 9; 

    mat_vec_dot(out, M, v);

    // Expected out = M*v = [1*7+2*8+3*9, 4*7+5*8+6*9]
    assert(out[0] == 50);
    assert(out[1] == 122);

    mat_destroy(M);
    vec_destroy(v);
    vec_destroy(out);
}

void test_matT_vec_dot() {
    mat_t M = mat_new(2, 3);
    vec_t v = vec_new(2);
    vec_t out = vec_new(3);

    // M = [[1,2,3],[4,5,6]]
    M[0][0] = 1; M[0][1] = 2; M[0][2] = 3;
    M[1][0] = 4; M[1][1] = 5; M[1][2] = 6;
    // v = [7,8]
    v[0] = 7; v[1] = 8;

    matT_vec_dot(out, M, v);

    // Expected out = M^T * v
    // [1*7+4*8, 2*7+5*8, 3*7+6*8] = [39,54,69]
    assert(out[0] == 39);
    assert(out[1] == 54);
    assert(out[2] == 69);

    mat_destroy(M);
    vec_destroy(v);
    vec_destroy(out);
}

int main() {
    test_vecT_vec_dot();
    test_mat_vec_dot();
    test_matT_vec_dot();
    printf("All tests passed!\n");
    return 0;
}
