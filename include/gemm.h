#ifndef GEMM_H
#define GEMM_H
#include <stdint.h>

void gemm_16(int M, int N, int K,
             int16_t* A, int16_t* B, int16_t* C);
void gemm_32(int M, int N, int K,
             float* A, float* B, float* C);

#endif
