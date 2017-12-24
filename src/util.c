#include <stdlib.h>
#include <malloc.h>
#include <time.h>
#include "util.h"
#include <stdio.h>

#define VRU_ENABLE

#ifdef VRU_ENABLE
// because gcc complains about shifting without L
#define VRU_SWITCH 0x8000000000000000
#else
#define VRU_SWITCH 0x0
#endif

#define VCFG(nvvd, nvvw, nvvh, nvp) \
  (((nvvd) & 0x1ff) | \
  (((nvp) & 0x1f) << 9) | \
  (((nvvw) & 0x1ff) << 14) | \
  (((nvvh) & 0x1ff) << 23) | \
  (VRU_SWITCH))

void hwacha_init() {
  asm volatile ("lw t0, vsetvlen" : : : "t0");
  
}
  

int __attribute__((optimize("O0"))) rdcycle() {
    int out = 0;
    asm("rdcycle %0" : "=r" (out));
    return out;
}

int __attribute__((optimize("O0"))) rdinstret() {
    int out = 0;
    asm("rdinstret %0" : "=r" (out));
    return out;
}

void* __attribute__((optimize("O0"))) safe_malloc(int size) {
    void* ptr = memalign(16, size);
    for (int i = 0; i < size / 4; i += (1 << 10)) {
        ((int*)ptr)[i] = 1;
    }
    return ptr;
}

void printfloatmatrix(int channels, int width, int height, float* M) {
    printf("\n");
    for (int c = 0; c < channels; c++) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                printf("%.3f\t", M[c*height*width+i*width+j]);
            }
            printf("\n");
        }
        printf("-----\n");
    }
}
void printintmatrix(int channels, int width, int height, int* M) {
    printf("\n");
    for (int c = 0; c < channels; c++) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                printf("%d\t", M[c*height*width+i*width+j]);
            }
            printf("\n");
        }
        printf("-----\n");
    }
}
void printint16matrix(int channels, int width, int height, int16_t* M) {
    printf("\n");
    for (int c = 0; c < channels; c++) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                printf("%hu\t", M[c*height*width+i*width+j]);
            }
            printf("\n");
        }
        printf("-----\n");
    }
}

void fill_seq_32(float* p, int n, int mode) {
    for (int i = 0; i < n; i++) {
        if (mode == 0) {
            p[i] = i;
        } else if (mode == 1) {
            p[i] = (float)rand() / (float)(RAND_MAX);
        } else if (mode == 2) {
            p[i] = 1;
        }
    }
}


void fill_seq_16(int16_t* p, int n, int mode) {
    for (int i = 0; i < n; i++) {
        if (mode == 0) {
            p[i] = i;
        } else if (mode == 1) {
          float f = (float)rand() / (float)RAND_MAX;
          cvt_32_16(&f, &p[i], 1);
        } else if (mode == 2) {
            p[i] = 1;
        }
    }
}

void setvcfg(int nd, int nw, int nh, int np) {
    int cfg = VCFG(nd, nw, nh, np);
    asm volatile ("vsetcfg %0"
                  :
                  : "r" (cfg));
}

int setvlen(int vlen) {
    int consumed;
    asm volatile ("vsetvl %0, %1"
                  : "=r" (consumed)
                  : "r" (vlen));
    asm volatile ("la t0, vsetvlen" : : : "t0");
    asm volatile ("vf 0(t0)");
    asm volatile ("fence");
    return consumed;
}

void memcpy_16(int16_t* src, int16_t* dest, int len)
{
  setvcfg(0, 0, 1, 1);
  for (int i = 0; i < len; ) {
    int consumed = setvlen(len - i);
    asm volatile ("vmca va0, %0"
                  :
                  : "r" (&src[i]));
    asm volatile ("vmca va1, %0"
                  :
                  : "r" (&dest[i]));
    asm volatile ("la t0, vmemcpy_16"
                  :
                  :
                  : "t0");
    asm volatile ("lw t1, 0(t0)");
    asm volatile ("vf 0(t0)");
    i += consumed;
  }
  asm volatile ("fence");
}

void memcpy_32(float* src, float* dest, int len)
{
  setvcfg(0, 1, 0, 1);
  for (int i = 0; i < len; ) {
    int consumed = setvlen(len - i);
    asm volatile ("vmca va0, %0"
                  :
                  : "r" (&src[i]));
    asm volatile ("vmca va1, %0"
                  :
                  : "r" (&dest[i]));
    asm volatile ("la t0, vmemcpy_32"
                  :
                  :
                  : "t0");
    asm volatile ("lw t1, 0(t0)");
    asm volatile ("vf 0(t0)");
    i += consumed;
  }
  asm volatile ("fence");
}

void cvt_32_16(float* src, int16_t* dest, int len)
{
  setvcfg(0, 1, 1, 1);
  for (int i = 0; i < len; ) {
    int consumed = setvlen(len - i);
    asm volatile ("vmca va0, %0"
                  :
                  : "r" (&src[i]));
    asm volatile ("vmca va1, %0"
                  :
                  : "r" (&dest[i]));
    asm volatile ("la t0, vcvt_32_16"
                  :
                  :
                  : "t0");
    asm volatile ("lw t1, 0(t0)");
    asm volatile ("vf 0(t0)");
    i += consumed;
  }
  asm volatile ("fence");
}

void cvt_16_32(int16_t* src, float* dest, int len)
{
  setvcfg(0, 1, 1, 1);
  for (int i = 0; i < len; )
    {
      int consumed = setvlen(len - i);
      asm volatile ("vmca va0, %0"
                    :
                    : "r" (&src[i]));
      asm volatile ("vmca va1, %0"
                    :
                    : "r" (&dest[i]));
      asm volatile ("la t0, vcvt_16_32"
                    :
                    :
                    : "t0");
      asm volatile ("lw t1, 0(t0)");
      asm volatile ("vf 0(t0)");
      i += consumed;
    }
  asm volatile ("fence");
}


void gather_16(const int* id, const int16_t* src, int16_t* dest, int len) {
    setvcfg(0, 1, 1, 2);
    asm volatile ("la t0, vgather_16" : : : "t0");
    asm volatile ("lw t1, 0(t0)");
    for (int i = 0; i < len; ) {
        int consumed = setvlen(len - i);

        asm volatile ("vmcs vs1, %0"
                      :
                      : "r" (&src[0]));
        asm volatile ("vmca va1, %0"
                      :
                      : "r" (&id[i]));
        asm volatile ("vmca va2, %0"
                      :
                      : "r" (&dest[i]));
        asm volatile ("la t0, vgather_16"
                      :
                      :
                      : "t0");
        asm volatile ("vf 0(t0)");
        i += consumed;
    }
    asm volatile ("fence");
}


void fill_16(int N, float ALPHA, int16_t * X)
{
   int i;
   setvcfg(0, 0, 1, 1);
   asm volatile ("vmcs vs1, %0"
                 :
                 : "r" (ALPHA));
   for (i = 0; i < N; )
     {
       int consumed = setvlen (N - i);
       asm volatile ("vmca va0, %0"
                     :
                     : "r" (&X[i]));
       asm volatile ("la t0, vfill_16"
                     :
                     :
                     : "t0");
       asm volatile ("lw t1, 0(t0)");
       asm volatile ("vf 0(t0)");
       i += consumed;
     }
   asm volatile ("fence");
}


void normalize_16(int16_t *x, int16_t *mean, int16_t *variance, int filters, int spatial)
{
    int f, i;
    setvcfg(0, 0, 1, 1);
    asm volatile ("vmcs vs3, %0" : : "r" (0.000001f));
    for(f = 0; f < filters; ++f)
      {
        asm volatile ("vmcs vs1, %0" : : "r" (mean[f]));
        asm volatile ("vmcs vs2, %0" : : "r" (variance[f]));
        for (i = 0; i < spatial ;)
          {
            int consumed = setvlen(spatial - i);
            asm volatile ("vmca va0, %0" : : "r" (&x[f*spatial + i]));
            asm volatile ("la t0, vnormalize_16" : : : "t0");
            asm volatile ("lw t1, 0(t0)");
            asm volatile ("vf 0(t0)");
            i += consumed;
          }
    }
    asm volatile ("fence");
}

void scale_16(int16_t* x, int16_t scale, int size)
{
  setvcfg(0, 0, 1, 1);
  for (int i = 0; i < size; )
    {
      int consumed = setvlen(size - i);
      asm volatile ("vmca va0, %0"
                    :
                    : "r" (&x[i]));
      asm volatile ("vmcs vs1, %0"
                    :
                    : "r" (scale));
      asm volatile ("la t0, vscale_16"
                    :
                    :
                    : "t0");
      asm volatile ("lw t1, 0(t0)");
      asm volatile ("vf 0(t0)");
      i += consumed;
    }
  asm volatile ("fence");

}

void add_16(int16_t* x, int16_t y, int size)
{
  setvcfg(0, 0, 1, 1);
  for (int i = 0; i < size; )
    {
      int consumed = setvlen(size - i);
      asm volatile ("vmca va0, %0"
                    :
                    : "r" (&x[i]));
      asm volatile ("vmcs vs1, %0"
                    :
                    : "r" (y));
      asm volatile ("la t0, vadd_16"
                    :
                    :
                    : "t0");
      asm volatile ("lw t1, 0(t0)");
      asm volatile ("vf 0(t0)");
      i += consumed;
    }
  asm volatile ("fence");

}
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
