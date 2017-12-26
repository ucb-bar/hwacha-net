#include "layer.h"
#include "util.h"
#include <math.h>
#include <stdint.h>

int conv_out_width(struct layer l)
{
  return (l.w + 2*l.pad - l.size) / l.stride + 1;
}
int conv_out_height(struct layer l)
{
  return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

void im2col_id(struct layer* l, int size)
{
  int height_col = l->output_h;
  int width_col = l->output_w;
  int height = l->h;
  int width = l->w;
  int ksize = l->size;
  int pad = l->pad;
  int stride = l->stride;
  int output_shape = height_col * width_col * ksize * ksize;

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
                          *(col_id_it++) = (im_ptr + input_row*width + input_col)*size;
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
  for (int c = 0; c < l->c; c++) {
    gather_16(l->indices, src + srcblock*c, b + destblock*c, destblock);
  }
  //gather_16(l->indices, src, b, l->output_h*l->output_w*l->size*l->size*l->c);
  gemm_16(m,n,k,a,b,c);
  //printf("%hu %hu %hu\n", dest[80000], dest[80001], dest[80002]);
}

void convolutional_precomp_forward_32(struct layer* l, float* src, float* dest, float* workspace)
{
  fill_32(l->output_h*l->output_w*l->output_c, 0, dest);
}
