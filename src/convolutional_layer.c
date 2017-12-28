#include "layer.h"
#include "util.h"
#include "gemm.h"
#include <math.h>
#include <stdint.h>
#define ENTRYCOUNT (5)
struct indices_entry
{
  int stride;
  int pad;
  int ksize;
  int height;
  int width;
  int* ptr;
};
struct indices_entry lookup_table[ENTRYCOUNT];
int current_count = 0;
int conv_out_width(struct layer* l)
{
  return (l->w + 2*l->pad - l->size) / l->stride + 1;
}
int conv_out_height(struct layer* l)
{
  return (l->h + 2*l->pad - l->size) / l->stride + 1;
}

void im2col_id(struct layer* l)
{
  int height_col = l->output_h;
  int width_col = l->output_w;
  int height = l->h;
  int width = l->w;
  int ksize = l->size;
  int pad = l->pad;
  int stride = l->stride;
  int output_shape = height_col * width_col * ksize * ksize;

  if (ksize == 1 && pad == 0 && stride == 1)
    {
      l->indices = NULL;
      return;
    }
  for (int i = 0; i < current_count; i++)
    {
      struct indices_entry* ie = &lookup_table[i];
      if (ie->stride == stride && ie->pad == pad && ie->ksize == ksize &&
          ie->height == height && ie->width == width)
        {
          //printf("found\n");
          l->indices = ie->ptr;
          return;
        }
    }

  int* id = safe_malloc (output_shape * sizeof(int));

  int* col_id_it = id;
  int im_ptr = 0;
  //printf("%d\n", size);
  for (int kernel_row = 0; kernel_row < ksize; kernel_row++)
    {
      for (int kernel_col = 0; kernel_col < ksize; kernel_col++)
        {
          int input_row = kernel_row - pad;
          for (int output_row = height_col; output_row; output_row--)
            {
              if (input_row < 0 || input_row >= height)
                {
                  for (int output_col = width_col; output_col; output_col--)
                    {
                      *(col_id_it++) = -1;
                    }
                }
              else
                {
                  int input_col = -pad + kernel_col;
                  for (int output_col = width_col; output_col; output_col--)
                    {
                      if (input_col >= 0 && input_col < width)
                        {
                          *(col_id_it++) = (im_ptr + input_row*width + input_col)*l->prec;
                        }
                      else
                        {
                          *(col_id_it++) = -1;
                        }
                      input_col += stride;
                    }
                }
              input_row += stride;
            }
        }
    }
  l->indices = id;
  struct indices_entry ie = {stride, pad, ksize, height, width, id};
  if (current_count < ENTRYCOUNT)
    lookup_table[current_count++] = ie;
  
}
float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;
    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}


void im2col_32(float* data_im,
               int channels,  int height,  int width,
               int ksize,  int stride, int pad, float* data_col) 
{
  int c,h,w;
  int height_col = (height + 2*pad - ksize) / stride + 1;
  int width_col = (width + 2*pad - ksize) / stride + 1;
  
  int channels_col = channels * ksize * ksize;
  for (c = 0; c < channels_col; ++c) {
    int w_offset = c % ksize;
    int h_offset = (c / ksize) % ksize;
    int c_im = c / ksize / ksize;
    for (h = 0; h < height_col; ++h) {
      for (w = 0; w < width_col; ++w) {
        int im_row = h_offset + h * stride;
        int im_col = w_offset + w * stride;
        int col_index = (c * height_col + h) * width_col + w;
        data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                                               im_row, im_col, c_im, pad);
      }
    }
  }
}


void convolutional_precomp_forward_16(struct layer* l, int16_t* src, int16_t* dest, int16_t* workspace)
{
  fill_16(l->output_h*l->output_w*l->output_c, 0, dest);
  int m = l->n;
  int k = l->size*l->size*l->c;
  int n = l->output_w*l->output_h;

  int16_t *a = l->weights_16;
  int16_t *b = workspace;
  int16_t *c = dest;

  int srcblock = l->h*l->w;
  int destblock = l->output_h*l->output_w*l->size*l->size;
  if (l->indices)
    {
      for (int c = 0; c < l->c; c++)
        gather_16(l->indices, src + srcblock*c, b + destblock*c, destblock);
      //gather_16(l->indices, src, b, l->output_h*l->output_w*l->size*l->size*l->c);
      gemm_16(m,n,k,a,b,c);
      //printf("%hu %hu %hu\n", dest[80000], dest[80001], dest[80002]);
    }
  else
    {
      gemm_16(m,n,k,a,src,c);
    }
}

void convolutional_precomp_forward_32(struct layer* l, float* src, float* dest, float* workspace)
{
  fill_32(l->output_h*l->output_w*l->output_c, 0, dest);
  int m = l->n;
  int k = l->size*l->size*l->c;
  int n = l->output_w*l->output_h;

  float *a = l->weights_32;
  float *b = workspace;
  float *c = dest;
  int srcblock = l->h*l->w;
  int destblock = l->output_h*l->output_w*l->size*l->size;
  if (l->indices)
    {
      for (int c = 0; c < l->c; c++)
        gather_32(l->indices, src + srcblock*c, b + destblock*c, destblock);
      gemm_32(m,n,k,a,b,c);
    }
  else
    {
      gemm_32(m,n,k,a,src,c);
    }
  //printf("%.3f conv\n", dest[0]);
}

void convolutional_forward_32(struct layer* l, float* src, float* dest, float* workspace)
{
  fill_32(l->output_h*l->output_w*l->output_c, 0, dest);

  int m = l->n;
  int k = l->size*l->size*l->c;
  int n = l->output_w*l->output_h;
  float *a = l->weights_32;
  float *b = workspace;
  float *c = dest;
  
  im2col_32(src, l->c, l->h, l->w, l->size, l->stride, l->pad, b);
  gemm_32(m,n,k,a,b,c);
  
  //printf("%.3f %.3f %.3f %.3f %.3f %.3f\n", dest[0], dest[1], dest[2], l->weights_32[0], l->weights_32[1], l->weights_32[2]);
}
