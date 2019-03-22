#include "gemm.h"
#include "util.h"
#include <stdio.h>

void gemm_16(int M, int N, int K,
             int16_t* A, int16_t* B, int16_t* C)
{
  int lda = K;
  int ldb = N;
  int ldc = N;
  setvcfg(0, 0, 8, 1);

  void * pre_vfblockaddr;
  void * pre_edge_vfblockaddr;
  void * main_vfblockaddr;
  void * main_edge0_vfblockaddr;
  void * main_edge1_vfblockaddr;
  void * post_vfblockaddr;
  void * post_edge_vfblockaddr;
  asm volatile ("la %0, hgemm_opt_v_4_4_pre" : "=r" (pre_vfblockaddr));
  asm volatile ("lw t0, 0(%0)" : : "r" (pre_vfblockaddr) : "t0");
  asm volatile ("la %0, hgemm_opt_v_4_4_pre_edge" : "=r" (pre_edge_vfblockaddr));
  asm volatile ("lw t0, 0(%0)" : : "r" (pre_edge_vfblockaddr) : "t0");
  asm volatile ("la %0, hgemm_opt_v_4_4" : "=r" (main_vfblockaddr));
  asm volatile ("lw t0, 0(%0)" : : "r" (main_vfblockaddr) : "t0");
  asm volatile ("la %0, hgemm_opt_v_4_4_edge0" : "=r" (main_edge0_vfblockaddr));
  asm volatile ("lw t0, 0(%0)" : : "r" (main_edge0_vfblockaddr) : "t0");
  asm volatile ("la %0, hgemm_opt_v_4_4_edge1" : "=r" (main_edge1_vfblockaddr));
  asm volatile ("lw t0, 0(%0)" : : "r" (main_edge1_vfblockaddr) : "t0");
  asm volatile ("la %0, hgemm_opt_v_4_4_post" : "=r" (post_vfblockaddr));
  asm volatile ("lw t0, 0(%0)" : : "r" (post_vfblockaddr) : "t0");
  asm volatile ("la %0, hgemm_opt_v_4_4_post_edge" : "=r" (post_edge_vfblockaddr));
  asm volatile ("lw t0, 0(%0)" : : "r" (post_edge_vfblockaddr) : "t0");
  int i;
  for (i = 0; i + 4 <= M; i+=4)
    {
      for (int k = 0; k < N; )
        {
          int consumed;
          int artificial = N - k;
          consumed = setvlen(artificial);

          // C rows 1, 2, 3, 4
          asm volatile ("vmca va0, %0" : : "r" (&C[i*ldc+k]));
          asm volatile ("vmca va1, %0" : : "r" (&C[(i+1)*ldc+k]));
          asm volatile ("vmca va2, %0" : : "r" (&C[(i+2)*ldc+k]));
          asm volatile ("vmca va3, %0" : : "r" (&C[(i+3)*ldc+k]));

          asm volatile ("vf 0(%0)" : : "r" (pre_vfblockaddr));
          int j;
          for (j = 0; j + 4 <= K; j+=4)
            {

              // B row 1, 2, 3, 4
              asm volatile ("vmca va4, %0" : : "r" (&B[j*ldb+k]));
              asm volatile ("vmca va5, %0" : : "r" (&B[(j+1)*ldb+k]));
              asm volatile ("vmca va6, %0" : : "r" (&B[(j+2)*ldb+k]));
              asm volatile ("vmca va7, %0" : : "r" (&B[(j+3)*ldb+k]));

              // A row 1, 2, 3, 4
              asm volatile ("vmcs vs1, %0\n"
                            "vmcs vs2, %1\n"
                            "vmcs vs3, %2\n"
                            "vmcs vs4, %3\n"

                            "vmcs vs5, %4\n"
                            "vmcs vs6, %5\n"
                            "vmcs vs7, %6\n"
                            "vmcs vs8, %7\n"

                            "vmcs vs9, %8\n"
                            "vmcs vs10, %9\n"
                            "vmcs vs11, %10\n"
                            "vmcs vs12, %11\n"

                            "vmcs vs13, %12\n"
                            "vmcs vs14, %13\n"
                            "vmcs vs15, %14\n"
                            "vmcs vs16, %15"
                            :
                            : "r" (A[j+(i+0)*lda+0]), "r" (A[j+(i+0)*lda+1]), "r" (A[j+(i+0)*lda+2]), "r" (A[j+(i+0)*lda+3]),
                              "r" (A[j+(i+1)*lda+0]), "r" (A[j+(i+1)*lda+1]), "r" (A[j+(i+1)*lda+2]), "r" (A[j+(i+1)*lda+3]),
                              "r" (A[j+(i+2)*lda+0]), "r" (A[j+(i+2)*lda+1]), "r" (A[j+(i+2)*lda+2]), "r" (A[j+(i+2)*lda+3]),
                              "r" (A[j+(i+3)*lda+0]), "r" (A[j+(i+3)*lda+1]), "r" (A[j+(i+3)*lda+2]), "r" (A[j+(i+3)*lda+3])
                            );

              asm volatile ("vf 0(%0)" : : "r" (main_vfblockaddr));
            }

          for ( ; j < K; j++)
            {
              asm volatile ("vmca va4, %0" : : "r" (&B[j*ldb+k]));

              asm volatile ("vmcs vs1, %0\n"
                            "vmcs vs5, %1\n"
                            "vmcs vs9, %2\n"
                            "vmcs vs13, %3\n"
                            :
                            : "r" (A[j+(i+0)*lda+0]),
                              "r" (A[j+(i+1)*lda+0]),
                              "r" (A[j+(i+2)*lda+0]),
                              "r" (A[j+(i+3)*lda+0])
                            );
              asm volatile ("vf 0(%0)" : : "r" (main_edge0_vfblockaddr));

            }
          asm volatile ("vf 0(%0)" : : "r" (post_vfblockaddr));
          k += consumed;
        }
    }

  for ( ; i < M; i++)
    {
      for (int k = 0; k < N; )
        {
          int consumed;
          int artificial = N - k;
          consumed = setvlen(artificial);
          asm volatile ("vmca va0, %0" : : "r" (&C[i*ldc+k]));

          asm volatile ("vf 0(%0)" : : "r" (pre_edge_vfblockaddr));

          for (int j = 0; j < K; j++)
            {
              asm volatile ("vmca va4, %0" : : "r" (&B[j*ldb+k]));
              asm volatile ("vmcs vs1, %0" : : "r" (A[j+(i+0)*lda+0]));
              asm volatile ("vf 0(%0)" : : "r" (main_edge1_vfblockaddr));
            }
          asm volatile ("vf 0(%0)" : : "r" (post_edge_vfblockaddr));
          k += consumed;
        }
    }
  asm volatile ("fence");
}

void gemm_32(int M, int N, int K,
             float* A, float* B, float* C)
{
#ifndef USE_SCALAR
  int lda = K;
  int ldb = N;
  int ldc = N;
  setvcfg(0, 8, 0, 1);

  void * pre_vfblockaddr;
  void * pre_edge_vfblockaddr;
  void * main_vfblockaddr;
  void * main_edge0_vfblockaddr;
  void * main_edge1_vfblockaddr;
  void * post_vfblockaddr;
  void * post_edge_vfblockaddr;
  asm volatile ("la %0, sgemm_opt_v_4_4_pre" : "=r" (pre_vfblockaddr));
  asm volatile ("lw t0, 0(%0)" : : "r" (pre_vfblockaddr) : "t0");
  asm volatile ("la %0, sgemm_opt_v_4_4_pre_edge" : "=r" (pre_edge_vfblockaddr));
  asm volatile ("lw t0, 0(%0)" : : "r" (pre_edge_vfblockaddr) : "t0");
  asm volatile ("la %0, sgemm_opt_v_4_4" : "=r" (main_vfblockaddr));
  asm volatile ("lw t0, 0(%0)" : : "r" (main_vfblockaddr) : "t0");
  asm volatile ("la %0, sgemm_opt_v_4_4_edge0" : "=r" (main_edge0_vfblockaddr));
  asm volatile ("lw t0, 0(%0)" : : "r" (main_edge0_vfblockaddr) : "t0");
  asm volatile ("la %0, sgemm_opt_v_4_4_edge1" : "=r" (main_edge1_vfblockaddr));
  asm volatile ("lw t0, 0(%0)" : : "r" (main_edge1_vfblockaddr) : "t0");
  asm volatile ("la %0, sgemm_opt_v_4_4_post" : "=r" (post_vfblockaddr));
  asm volatile ("lw t0, 0(%0)" : : "r" (post_vfblockaddr) : "t0");
  asm volatile ("la %0, sgemm_opt_v_4_4_post_edge" : "=r" (post_edge_vfblockaddr));
  asm volatile ("lw t0, 0(%0)" : : "r" (post_edge_vfblockaddr) : "t0");
  int i;
  for (i = 0; i + 4 <= M; i+=4)
    {
      for (int k = 0; k < N; )
        {
          int consumed;
          int artificial = N - k;
          consumed = setvlen(artificial);

          // C rows 1, 2, 3, 4
          asm volatile ("vmca va0, %0" : : "r" (&C[i*ldc+k]));
          asm volatile ("vmca va1, %0" : : "r" (&C[(i+1)*ldc+k]));
          asm volatile ("vmca va2, %0" : : "r" (&C[(i+2)*ldc+k]));
          asm volatile ("vmca va3, %0" : : "r" (&C[(i+3)*ldc+k]));

          asm volatile ("vf 0(%0)" : : "r" (pre_vfblockaddr));
          int j = 0;
          /* for (j = 0; j + 4 <= K; j+=4) */
          /*   { */

          /*     // B row 1, 2, 3, 4 */
          /*     asm volatile ("vmca va4, %0" : : "r" (&B[j*ldb+k])); */
          /*     asm volatile ("vmca va5, %0" : : "r" (&B[(j+1)*ldb+k])); */
          /*     asm volatile ("vmca va6, %0" : : "r" (&B[(j+2)*ldb+k])); */
          /*     asm volatile ("vmca va7, %0" : : "r" (&B[(j+3)*ldb+k])); */

          /*     // A row 1, 2, 3, 4 */
          /*     asm volatile ("vmcs vs1, %0\n" */
          /*                   "vmcs vs2, %1\n" */
          /*                   "vmcs vs3, %2\n" */
          /*                   "vmcs vs4, %3\n" */

          /*                   "vmcs vs5, %4\n" */
          /*                   "vmcs vs6, %5\n" */
          /*                   "vmcs vs7, %6\n" */
          /*                   "vmcs vs8, %7\n" */

          /*                   "vmcs vs9, %8\n" */
          /*                   "vmcs vs10, %9\n" */
          /*                   "vmcs vs11, %10\n" */
          /*                   "vmcs vs12, %11\n" */

          /*                   "vmcs vs13, %12\n" */
          /*                   "vmcs vs14, %13\n" */
          /*                   "vmcs vs15, %14\n" */
          /*                   "vmcs vs16, %15" */
          /*                   : */
          /*                   : "r" (A[j+(i+0)*lda+0]), "r" (A[j+(i+0)*lda+1]), "r" (A[j+(i+0)*lda+2]), "r" (A[j+(i+0)*lda+3]), */
          /*                     "r" (A[j+(i+1)*lda+0]), "r" (A[j+(i+1)*lda+1]), "r" (A[j+(i+1)*lda+2]), "r" (A[j+(i+1)*lda+3]), */
          /*                     "r" (A[j+(i+2)*lda+0]), "r" (A[j+(i+2)*lda+1]), "r" (A[j+(i+2)*lda+2]), "r" (A[j+(i+2)*lda+3]), */
          /*                     "r" (A[j+(i+3)*lda+0]), "r" (A[j+(i+3)*lda+1]), "r" (A[j+(i+3)*lda+2]), "r" (A[j+(i+3)*lda+3]) */
          /*                   ); */

          /*     asm volatile ("vf 0(%0)" : : "r" (main_vfblockaddr)); */
          /*   } */

          for ( ; j < K; j++)
            {
              asm volatile ("vmca va4, %0" : : "r" (&B[j*ldb+k]));

              asm volatile ("vmcs vs1, %0\n"
                            "vmcs vs5, %1\n"
                            "vmcs vs9, %2\n"
                            "vmcs vs13, %3\n"
                            :
                            : "r" (A[j+(i+0)*lda+0]),
                              "r" (A[j+(i+1)*lda+0]),
                              "r" (A[j+(i+2)*lda+0]),
                              "r" (A[j+(i+3)*lda+0])
                            );
              asm volatile ("vf 0(%0)" : : "r" (main_edge0_vfblockaddr));

            }
          asm volatile ("vf 0(%0)" : : "r" (post_vfblockaddr));
          k += consumed;
        }
    }

  for ( ; i < M; i++)
    {
      for (int k = 0; k < N; )
        {
          int consumed;
          int artificial = N - k;
          consumed = setvlen(artificial);
          asm volatile ("vmca va0, %0" : : "r" (&C[i*ldc+k]));

          asm volatile ("vf 0(%0)" : : "r" (pre_edge_vfblockaddr));

          for (int j = 0; j < K; j++)
            {
              asm volatile ("vmca va4, %0" : : "r" (&B[j*ldb+k]));
              asm volatile ("vmcs vs1, %0" : : "r" (A[j+(i+0)*lda+0]));
              asm volatile ("vf 0(%0)" : : "r" (main_edge1_vfblockaddr));
            }
          asm volatile ("vf 0(%0)" : : "r" (post_edge_vfblockaddr));
          k += consumed;
        }
    }
  asm volatile ("fence");
#else
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++)
      for (int k = 0; k < K; k++)
        C[m*N+n] += A[m*K+k]*B[k*N+n];
#endif
}


void gemm_encoded_32(int M, int N, int K,
                     unsigned char* A, float* B, float* C,
                     float* codebook)
{
#ifndef USE_SCALAR
  int lda = K;
  int ldb = N;
  int ldc = N;
  setvcfg(0, 5, 0, 1);

  void * pre_vfblockaddr;
  void * pre_edge_vfblockaddr;
  void * main_vfblockaddr;
  void * main_edge0_vfblockaddr;
  void * main_edge1_vfblockaddr;
  void * post_vfblockaddr;
  void * post_edge_vfblockaddr;
  asm volatile ("la %0, sgemm_opt_v_4_4_pre" : "=r" (pre_vfblockaddr));
  asm volatile ("lw t0, 0(%0)" : : "r" (pre_vfblockaddr) : "t0");
  asm volatile ("la %0, sgemm_opt_v_4_4_pre_edge" : "=r" (pre_edge_vfblockaddr));
  asm volatile ("lw t0, 0(%0)" : : "r" (pre_edge_vfblockaddr) : "t0");
  asm volatile ("la %0, sgemm_opt_v_4_4" : "=r" (main_vfblockaddr));
  asm volatile ("lw t0, 0(%0)" : : "r" (main_vfblockaddr) : "t0");
  asm volatile ("la %0, sgemm_opt_v_4_4_edge0" : "=r" (main_edge0_vfblockaddr));
  asm volatile ("lw t0, 0(%0)" : : "r" (main_edge0_vfblockaddr) : "t0");
  asm volatile ("la %0, sgemm_opt_v_4_4_edge1" : "=r" (main_edge1_vfblockaddr));
  asm volatile ("lw t0, 0(%0)" : : "r" (main_edge1_vfblockaddr) : "t0");
  asm volatile ("la %0, sgemm_opt_v_4_4_post" : "=r" (post_vfblockaddr));
  asm volatile ("lw t0, 0(%0)" : : "r" (post_vfblockaddr) : "t0");
  asm volatile ("la %0, sgemm_opt_v_4_4_post_edge" : "=r" (post_edge_vfblockaddr));
  asm volatile ("lw t0, 0(%0)" : : "r" (post_edge_vfblockaddr) : "t0");
  int i;
  for (i = 0; i + 4 <= M; i+=4)
    {
      for (int k = 0; k < N; )
        {
          int consumed;
          int artificial = N - k;
          consumed = setvlen(artificial);

          // C rows 1, 2, 3, 4
          asm volatile ("vmca va0, %0" : : "r" (&C[i*ldc+k]));
          asm volatile ("vmca va1, %0" : : "r" (&C[(i+1)*ldc+k]));
          asm volatile ("vmca va2, %0" : : "r" (&C[(i+2)*ldc+k]));
          asm volatile ("vmca va3, %0" : : "r" (&C[(i+3)*ldc+k]));

          asm volatile ("vf 0(%0)" : : "r" (pre_vfblockaddr));
          int j = 0;
          for ( ; j < K; j++)
            {
              asm volatile ("vmca va4, %0" : : "r" (&B[j*ldb+k]));
              asm volatile ("vmcs vs1, %0\n"
                            "vmcs vs5, %1\n"
                            "vmcs vs9, %2\n"
                            "vmcs vs13, %3\n"
                            :
                            : "r" (codebook[A[j+(i+0)*lda+0]]),
                              "r" (codebook[A[j+(i+1)*lda+0]]),
                              "r" (codebook[A[j+(i+2)*lda+0]]),
                              "r" (codebook[A[j+(i+3)*lda+0]])
                            );
              asm volatile ("vf 0(%0)" : : "r" (main_edge0_vfblockaddr));

            }
          asm volatile ("vf 0(%0)" : : "r" (post_vfblockaddr));
          k += consumed;
        }
    }

  for ( ; i < M; i++)
    {
      for (int k = 0; k < N; )
        {
          int consumed;
          int artificial = N - k;
          consumed = setvlen(artificial);
          asm volatile ("vmca va0, %0" : : "r" (&C[i*ldc+k]));

          asm volatile ("vf 0(%0)" : : "r" (pre_edge_vfblockaddr));

          for (int j = 0; j < K; j++)
            {
              asm volatile ("vmca va4, %0" : : "r" (&B[j*ldb+k]));
              asm volatile ("vmcs vs1, %0" : : "r" (codebook[A[j+(i+0)*lda+0]]));
              asm volatile ("vf 0(%0)" : : "r" (main_edge1_vfblockaddr));
            }
          asm volatile ("vf 0(%0)" : : "r" (post_edge_vfblockaddr));
          k += consumed;
        }
    }
  asm volatile ("fence");
#else
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++)
      for (int k = 0; k < K; k++)
        C[m*N+n] += codebook[A[m*K+k]]*B[k*N+n];
#endif
}

void gemm_encoded_compressed_32(int M, int N, int K,
                                uint8_t* indices, uint8_t* indptr, uint8_t* data,
                                float* B, float* C,
                                float* codebook)
{
  int nrows = M;
  int ncols = K;
  int row = 0;
  int indptrptr = 1;
  int dataptr = indptr[0];
  int ldb = N;
  int ldc = N;

#ifndef USE_SCALAR
  //printf("%d %d %d\n", M, N, K);
  //int lda = K;
  setvcfg(0, 5, 0, 1);
  void * pre_vfblockaddr;
  void * pre_edge_vfblockaddr;
  void * main_edge0_vfblockaddr;
  void * main_edge1_vfblockaddr;
  void * post_vfblockaddr;
  void * post_edge_vfblockaddr;
  asm volatile ("la %0, sgemm_opt_v_4_4_pre" : "=r" (pre_vfblockaddr));
  asm volatile ("lw t0, 0(%0)" : : "r" (pre_vfblockaddr) : "t0");
  asm volatile ("la %0, sgemm_opt_v_4_4_pre_edge" : "=r" (pre_edge_vfblockaddr));
  asm volatile ("lw t0, 0(%0)" : : "r" (pre_edge_vfblockaddr) : "t0");
  asm volatile ("la %0, sgemm_opt_v_4_4_edge0" : "=r" (main_edge0_vfblockaddr));
  asm volatile ("lw t0, 0(%0)" : : "r" (main_edge0_vfblockaddr) : "t0");
  asm volatile ("la %0, sgemm_opt_v_4_4_edge1" : "=r" (main_edge1_vfblockaddr));
  asm volatile ("lw t0, 0(%0)" : : "r" (main_edge1_vfblockaddr) : "t0");
  asm volatile ("la %0, sgemm_opt_v_4_4_post" : "=r" (post_vfblockaddr));
  asm volatile ("lw t0, 0(%0)" : : "r" (post_vfblockaddr) : "t0");
  asm volatile ("la %0, sgemm_opt_v_4_4_post_edge" : "=r" (post_edge_vfblockaddr));
  asm volatile ("lw t0, 0(%0)" : : "r" (post_edge_vfblockaddr) : "t0");

  for ( ; row + 4 <= nrows; row += 4)
    {
      int rowa = dataptr;
      int rowb = rowa;
      while (indptr[indptrptr] == 255)
        {
          indptrptr++; rowb += 255;
        }
      rowb += indptr[indptrptr++];

      int rowc = rowb;
      while (indptr[indptrptr] == 255)
        {
          indptrptr++; rowc += 255;
        }
      rowc += indptr[indptrptr++];

      int rowd = rowc;
      while (indptr[indptrptr] == 255)
        {
          indptrptr++; rowd += 255;
        }
      rowd += indptr[indptrptr++];

      int rowe = rowd;
      while (indptr[indptrptr] == 255)
        {
          indptrptr++; rowe += 255;
        }
      rowe += indptr[indptrptr++];

      int i = row;
      for (int k = 0; k < N; )
        {
          int consumed;
          int artificial = N - k;
          consumed = setvlen(artificial);

          // C rows 1, 2, 3, 4
          asm volatile ("vmca va0, %0" : : "r" (&C[i*ldc+k]));
          asm volatile ("vmca va1, %0" : : "r" (&C[(i+1)*ldc+k]));
          asm volatile ("vmca va2, %0" : : "r" (&C[(i+2)*ldc+k]));
          asm volatile ("vmca va3, %0" : : "r" (&C[(i+3)*ldc+k]));

          asm volatile ("vf 0(%0)" : : "r" (pre_vfblockaddr));

          int col = 0;
          int rowaptr = rowa;
          int rowbptr = rowb;
          int rowcptr = rowc;
          int rowdptr = rowd;
          int cola = indices[rowaptr];
          int colb = indices[rowbptr];
          int colc = indices[rowcptr];
          int cold = indices[rowdptr];
          for ( ; col < ncols; col++)
            {
              
              int used = 0;
              asm volatile ("vmcs vs1, %0\n" : : "r" (0));
              asm volatile ("vmcs vs5, %0\n" : : "r" (0));
              asm volatile ("vmcs vs9, %0\n" : : "r" (0));
              asm volatile ("vmcs vs13, %0\n" : : "r" (0));
              if (rowaptr < rowb && cola == col)
                {
                  asm volatile ("vmcs vs1, %0\n" : : "r" (codebook[data[rowaptr++]]));
                  cola += indices[rowaptr];
                  used = 1;
                }
              if (rowbptr < rowc && colb == col)
                {
                  asm volatile ("vmcs vs5, %0\n" : : "r" (codebook[data[rowbptr++]]));
                  colb += indices[rowbptr];
                  used = 1;
                }
              if (rowcptr < rowd && colc == col)
                {
                  asm volatile ("vmcs vs9, %0\n" : : "r" (codebook[data[rowcptr++]]));
                  colc += indices[rowcptr];
                  used = 1;
                }
              if (rowdptr < rowe && cold == col)
                {
                  asm volatile ("vmcs vs13, %0\n" : : "r" (codebook[data[rowdptr++]]));
                  cold += indices[rowdptr];
                  used = 1;
                }
              if (used)
                {
                  asm volatile ("vmca va4, %0" : : "r" (&B[col*ldb+k]));
                  asm volatile ("vf 0(%0)" : : "r" (main_edge0_vfblockaddr));
                } 
            }
          asm volatile ("vf 0(%0)" : : "r" (post_vfblockaddr));
          k += consumed;
        }
      dataptr = rowe;
    }
  for ( ; row < nrows; row ++)
    {
      int aentries = 0;
      while (indptr[indptrptr] == 255)
        {
          aentries += 255;
          indptrptr++;
        }
      aentries += indptr[indptrptr++];
      for (int k = 0; k < N; )
        {
          int consumed;
          int artificial = N - k;
          consumed = setvlen(artificial);
          int i = row;
          asm volatile ("vmca va0, %0" : : "r" (&C[i*ldc+k]));
          asm volatile ("vf 0(%0)" : : "r" (pre_edge_vfblockaddr));
          int col = indices[dataptr];
          int dataptrptr = dataptr;
          for (int j = 0; j < aentries; j++)
            {
              asm volatile ("vmca va4, %0" : : "r" (&B[col*ldb+k]));
              asm volatile ("vmcs vs1, %0" : : "r" (codebook[data[dataptrptr++]]));
              col += indices[dataptrptr];
              asm volatile ("vf 0(%0)" : : "r" (main_edge1_vfblockaddr));
            }
          asm volatile ("vf 0(%0)" : : "r" (post_edge_vfblockaddr));
          k += consumed;
        }
      dataptr += aentries;
    }
  asm volatile ("fence");
#else
  for ( ; row < nrows; row ++)
    {
      int aentries = 0;
      while (indptr[indptrptr] == 255)
        {
          aentries += 255;
          indptrptr++;
        }
      aentries += indptr[indptrptr++];
      for (int k = 0; k < N; k++)
        {
          int i = row;
          int col = indices[dataptr];
          int dataptrptr = dataptr;
          for (int j = 0; j < aentries; j++)
            {
              C[i*ldc + k] += codebook[data[dataptrptr++]] * B[col*ldb+k];
              col += indices[dataptrptr];
            }
        }
      dataptr += aentries;
    }
#endif
}
