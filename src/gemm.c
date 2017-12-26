#include "gemm.h"
#include "util.h"

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
