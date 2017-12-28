#include "layer.h"
#include "util.h"
#include <float.h>
#include <math.h>
#include <stdint.h>


void maxpool_darknet_id_16(struct layer* l)
{
  int h = l->output_h;
  int w = l->output_w;
  int c = l->output_c;
  int w_offset = -l->pad;
  int h_offset = -l->pad;
  int out_shape = l->output_h * l->output_w * l->output_c;
  l->indices = safe_malloc(out_shape * l->size * l->size * sizeof(int));
  for (int k = 0; k < c; k++)
    {
      for (int i = 0; i < h; ++i)
        {
          for (int j = 0; j < w; ++j)
            {
              int out_index = j + w*(i + h*k);
              for (int n = 0; n < l->size; ++n)
                {
                  for (int m = 0; m < l->size; ++m)
                    {
                      int cur_h = h_offset + i*l->stride + n;
                      int cur_w = w_offset + j*l->stride + m;
                      int index = cur_w + l->w*(cur_h + l->h*k);
                      int valid = (cur_h >= 0 && cur_h < l->h &&
                                   cur_w >= 0 && cur_w < l->w);
                      int* off = l->indices + out_shape * (n * l->size + m);
                      if (valid)
                        off[out_index] = index * sizeof(int16_t);
                      else
                        off[out_index] = -1;

                    }
                }
            }
        }
    }
}

void maxpool_darknet_precomp_forward(struct layer* l, int16_t* src, int16_t* dest)
{
  setvcfg(0, 1, 2, 2);
  int outputs = l->output_h * l->output_w * l->output_c;
  int16_t min = -1024;
  asm volatile ("vmcs vs1, %0" : : "r" (min));
  asm volatile ("vmcs vs2, %0" : : "r" (src));
  asm volatile ("la t0, vmax_16_init" : : : "t0");
  asm volatile ("lw t0, 0(t0)");
  asm volatile ("la t0, vmax_16_iter" : : : "t0");
  asm volatile ("lw t0, 0(t0)");
  asm volatile ("la t0, vmax_16_st" : : : "t0");
  asm volatile ("lw t0, 0(t0)");

  for (int i = 0; i < outputs; )
    {
      int consumed = setvlen(outputs - i);

      asm volatile ("vmca va0, %0" : : "r" (&dest[i]));
      asm volatile ("la t0, vmax_16_init" : : : "t0");
      asm volatile ("vf 0(t0)");
      for (int n = 0; n < l->size; ++n)
        {
          for (int m = 0; m < l->size; ++m)
            {
              int off = (n * l->size + m) * outputs;
              asm volatile ("vmca va1, %0" : : "r" (&l->indices[off + i]));
              asm volatile ("la t0, vmax_16_iter" : : : "t0");
              asm volatile ("vf 0(t0)");
            }
        }

      asm volatile ("la t0, vmax_16_st" : : : "t0");
      asm volatile ("vf 0(t0)");
      i += consumed;
    }
  asm volatile ("fence");
  memcpy_16(dest, src, outputs);
  printf("%hu %hu %hu %hu %hu maxpool\n", src[1000], src[2000], src[4000], src[16000], src[32000]);
}
void maxpool_darknet_forward_16(struct layer* l, int16_t* src, int16_t* dest)
{
   int i,j,k,m,n;
   //int outputs = l->output_h * l->output_w * l->output_c;
   //printf("%d outputs\n", outputs);
   int w_offset = -l->pad;
   int h_offset = -l->pad;
   int h = l->output_h;
   int w = l->output_w;
   int c = l->c;
   for(k = 0; k < c; ++k)
     {
       for(i = 0; i < h; ++i)
         {
           for(j = 0; j < w; ++j)
             {
               int out_index = j + w*(i + h*k);
               int16_t max = -1024;
               int max_i = -1;
               for(n = 0; n < l->size; ++n)
                 {
                   for(m = 0; m < l->size; ++m)
                     {
                       int cur_h = h_offset + i*l->stride + n;
                       int cur_w = w_offset + j*l->stride + m;
                       int index = cur_w + l->w*(cur_h + l->h*k);
                       int valid = (cur_h >= 0 && cur_h < l->h &&
                                    cur_w >= 0 && cur_w < l->w);
                       int16_t val = (valid != 0) ? src[index] : -1024;
                       int s = val > max;
                       if (val < 0 && max < 0)
                         s = ~s;
                       max_i = (s) ? index : max_i;
                       max   = (s) ? val   : max;
                     }
                 }
               src[out_index] = max;

             }
         }
     }
   //printf("%hu %hu %hu %hu %hu maxpool\n", src[1000], src[2000], src[4000], src[16000], src[32000]);
}

void maxpool_darknet_forward_32(struct layer* l, float* src, float* dest)
{
  int i,j,k,m,n;
  //int outputs = l->output_h * l->output_w * l->output_c;
  //printf("%d outputs\n", outputs);
  int w_offset = -l->pad;
  int h_offset = -l->pad;
  int h = l->output_h;
  int w = l->output_w;
  int c = l->c;
  for(k = 0; k < c; ++k)
    {
      for(i = 0; i < h; ++i)
        {
          for(j = 0; j < w; ++j)
            {
              int out_index = j + w*(i + h*k);
              float max = -FLT_MAX;
              for(n = 0; n < l->size; ++n)
                {
                  for(m = 0; m < l->size; ++m)
                    {
                      int cur_h = h_offset + i*l->stride + n;
                      int cur_w = w_offset + j*l->stride + m;
                      int index = cur_w + l->w*(cur_h + l->h*k);
                      int valid = (cur_h >= 0 && cur_h < l->h &&
                                   cur_w >= 0 && cur_w < l->w);
                      float val = (valid != 0) ? src[index] : -FLT_MAX;
                      int s = val > max;
                      max   = (s) ? val   : max;
                    }
                }
              src[out_index] = max;
            }
        }
    }
  //printf("%.3f maxpool\n", src[0]);
  //memcpy_32(dest, src, h*w*c);
  //printf("%hu %hu %hu %hu %hu maxpool\n", src[1000], src[2000], src[4000], src[16000], src[32000]);
}
