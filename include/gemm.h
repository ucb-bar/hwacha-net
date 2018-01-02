#ifndef GEMM_H
#define GEMM_H
#include <stdint.h>

void gemm_16(int M, int N, int K,
             int16_t* A, int16_t* B, int16_t* C);
void gemm_32(int M, int N, int K,
             float* A, float* B, float* C);
void gemm_encoded_32(int M, int N, int K,
                     unsigned char* A, float* B, float* C,
                     float* codebook);
void gemm_encoded_compressed_32(int M, int N, int K,
                                unsigned char* indices,
                                unsigned char* indptr,
                                unsigned char* dtaa,
                                float* B, float* C,
                                float* codebook);

#endif
