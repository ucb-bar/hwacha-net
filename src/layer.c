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

void im2col_id_16(struct layer* l)
{
  int height_col = l->output_h;
  int width_col = l->output_w;
  int height = l->h;
  int width = l->w;
  int channels = l->c;
  int ksize = l->size;
  int pad = l->pad;
  int stride = l->stride;
  int channel_size = height*width;
  int output_shape = height_col * width_col * ksize * ksize * channels;

  int* id = safe_malloc (output_shape * sizeof(int));

  int* col_id_it = id;
  int im_ptr = 0;

  for (int channel = channels; channel--; im_ptr += channel_size) {
    for (int kernel_row = 0; kernel_row < ksize; kernel_row++) {
      for (int kernel_col = 0; kernel_col < ksize; kernel_col++) {
        int input_row = kernel_row - pad;
        for (int output_row = height_col; output_row; output_row--) {
          if (input_row < 0 || input_row >= height) {
            for (int output_col = width_col; output_col; output_col--) {
              *(col_id_it++) = -1;
            }
          } else {
            int input_col = -pad + kernel_col;
            for (int output_col = width_col; output_col; output_col--) {
              if (input_col >= 0 && input_col < width) {
                *(col_id_it++) = (im_ptr + input_row*width + input_col)*sizeof(int16_t);
              } else {
                *(col_id_it++) = -1;
              }
              input_col += stride;
            }
          }
          input_row += stride;
        }
      }
    }
  }
  l->indices = id;
}

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


void setup_layers(struct layer* l1, struct layer* l2)
{
  l2->h = l1->output_h;
  l2->w = l1->output_w;
  l2->c = l1->output_c;

  l2->workspace_size = 0;
  l2->nweights = 0;
  l2->output_w = l2->w; l2->output_h = l2->h; l2->output_c = l2->c;
  //printf("%d %d %d \t %d %d %d ->", l2->h, l2->w, l2->c, l2->output_h, l2->output_w, l2->output_c);
  switch (l2->type)
    {
    case CONVOLUTIONAL:
      {
        l2->output_w = conv_out_width(*l2);
        l2->output_h = conv_out_height(*l2);
        l2->output_c = l2->n;
        l2->workspace_size = l2->output_h * l2->output_w * l2->c * l2->size * l2->size;
        l2->nweights = l2->size * l2->size * l2->c * l2->n;
        im2col_id_16(l2);
        break;
      }
    case MAXPOOL_DARKNET:
      {
        l2->output_w = (l2->w + 2*l2->pad)/l2->stride;
        l2->output_h = (l2->h + 2*l2->pad)/l2->stride;
        l2->output_c = l2->c;
        //maxpool_darknet_id_16(l2);
        break;
      }
    case BATCHNORM:
      {
        l2->nweights = 3*l2->output_c;
        break;
      }
    case BIAS:
      {
        l2->nweights = l2->c;
        break;
      }
    case REGION:
      {
        l2->output_c = l2->n*(l2->size + 4 + 1);
        break;
      }
    case MAXPOOL:
      {
        printf("ERROR\n");
        break;
      }
    default:
      {
        break;
      }
    }
  l2->input_size = l2->h * l2->c * l2->w;
  l2->output_size = l2->output_w * l2->output_h * l2->output_c;
  //printf("%d %d %d \t %d %d %d \n", l2->h, l2->w, l2->c, l2->output_h, l2->output_w, l2->output_c);

}

void load_layers(struct layer** layers, int n, FILE* fp)
{
  for (int i = 0; i < n; i++) {
    layer* l = layers[i];
    l->weights_16 = safe_malloc(l->nweights * sizeof(int16_t));
    fread(l->weights_16, sizeof(int16_t), l->nweights, fp);
    //if (l->nweights) printf("%hu\n", l->weights_16[0]);
  }
}

size_t max_size(struct layer** layers, int n)
{
  size_t max = 0;
  for (int i = 0; i < n; i++)
    max = MAX(max, MAX(layers[i]->input_size, layers[i]->output_size));
  return max;
}

size_t max_workspace(struct layer** layers, int n)
{
  size_t max = 0;
  for (int i = 0; i < n; i++)
    max = MAX(max, layers[i]->workspace_size);
  return max;
}

void convolutional_precomp_forward(struct layer* l, int16_t* src, int16_t* dest, int16_t* workspace)
{
  fill_16(l->output_h*l->output_w*l->output_c, 0, dest);
  int m = l->n;
  int k = l->size*l->size*l->c;
  int n = l->output_w*l->output_h;

  int16_t *a = l->weights_16;
  int16_t *b = workspace;
  int16_t *c = dest;

  gather_16(l->indices, src, b, l->output_h*l->output_w*l->size*l->size*l->c);
  gemm_16(m,n,k,a,b,c);
  printf("%hu %hu %hu\n", dest[80000], dest[80001], dest[80002]);

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
void maxpool_darknet_forward(struct layer* l, int16_t* src, int16_t* dest)
{
   int i,j,k,m,n;
   int outputs = l->output_h * l->output_w * l->output_c;
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
   return;


}
void batchnorm_forward(struct layer* l, int16_t* src, int16_t* dest)
{
  int16_t* scales = l->weights_16;
  int16_t* rolling_mean = l->weights_16 + l->output_c;
  int16_t* rolling_variance = l->weights_16 + 2 * l->output_c;

  normalize_16(src, rolling_mean, rolling_variance, l->output_c, l->output_h*l->output_w);


  for(int i = 0; i < l->output_c; ++i)
    scale_16 (&src[i*l->output_h*l->output_w], scales[i], l->output_h*l->output_w);

}
void bias_forward(struct layer* l, int16_t* src)
{
  for(int i = 0; i < l->output_c; ++i)
    add_16 (&src[i*l->output_h*l->output_w], l->weights_16[i], l->output_h*l->output_w);
}

void leaky_forward(struct layer* l, int16_t* src)
{
  setvcfg(0, 0, 1, 2);
  float a = 0.1f;
  asm volatile ("vmcs vs2, %0"
                :
                : "r" (a));
  int len = l->output_h*l->output_w*l->output_c;
  for (int i = 0; i < len; )
    {
      int consumed = setvlen(len - i);
      asm volatile ("vmca va0, %0"
                    :
                    : "r" (&src[i]));
      asm volatile ("la t0, vleaky_activate_16"
                    :
                    :
                    : "t0");
      asm volatile ("lw t1, 0(t0)");
      asm volatile ("vf 0(t0)");
      i += consumed;
    }
  asm volatile ("fence");

}

int entry_index(struct layer* l, int location, int entry)
{
    int n =   location / (l->w*l->h);
    int loc = location % (l->w*l->h);
    return n*l->w*l->h*(4+l->size+1) + entry*l->w*l->h + loc;
}

static inline float logistic_activate(float x){return 1./(1. + exp(-x));}

void logistic_16(int16_t* src, int size)
{
  float buf[1024];
  for (int i = 0; i < size; )
    {
      int consumed = 1024 < size - i ? 1024 : size - i;
      cvt_16_32 (&src[i], buf, consumed);
      for (int j = 0; j < consumed; j++)
        buf[j] = logistic_activate (buf[j]);
      cvt_32_16 (buf, &src[i], consumed);
      i += consumed;
    }
}
void softmax(float *input, int n, int stride, float *output)
{
    int i;
    float sum = 0;
    float largest = -99999999999.f;
    for(i = 0; i < n; ++i){
        if(input[i*stride] > largest) largest = input[i*stride];
    }
    for(i = 0; i < n; ++i){
        float e = exp(input[i*stride] - largest);
        sum += e;
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}

void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int stride, float *output)
{
  int g, b;
  for(b = 0; b < batch; ++b)
    {
      for(g = 0; g < groups; ++g)
        {
          softmax(input + b*batch_offset + g, n, stride, output + b*batch_offset + g);
        }
    }
}
void region_forward(struct layer* l, int16_t* src, int16_t* dest, int16_t* workspace)
{
  float* in_32 = (float*) workspace;
  float* out_32 = in_32 + l->w*l->h*l->c;
  int index;
  memcpy_16(src, dest, l->h*l->w*l->c);
  for(int n = 0; n < l->n; ++n)
    {
      index = entry_index(l, n*l->w*l->h, 0);
      logistic_16(dest+index, 2*l->w*l->h);
      index = entry_index(l, n*l->w*l->h, 4);
      logistic_16(dest+index, l->w*l->h);
    }

  cvt_16_32(dest, out_32, l->output_h*l->output_w*l->output_c);
  cvt_16_32(src, in_32, l->h*l->w*l->c);
  index = entry_index(l, 0, 4 + 1);
  softmax_cpu(in_32 + index, l->size, l->n, l->h*l->w*l->c/l->n, l->w*l->h, l->w*l->h, out_32 + index);
  memcpy_32(out_32, (float*) dest, l->output_h*l->output_w*l->output_c);
}

void layer_forward(struct layer* layer, int16_t* src, int16_t* dest, int16_t* workspace)
{
  size_t cycles = rdcycle();
  switch (layer->type)
    {
    case CONVOLUTIONAL: { convolutional_precomp_forward(layer, src, dest, workspace); break; }
    case MAXPOOL_DARKNET: { maxpool_darknet_forward(layer, src, dest); break; }
    case BATCHNORM: { batchnorm_forward(layer, src, dest); break; }
    case REGION: { region_forward(layer, src, dest, workspace); break; }
    case BIAS: { bias_forward(layer, src); break; }
    case LEAKY: { leaky_forward(layer, src); break; }
    default: { break; }
    }
  cycles = rdcycle() - cycles;
  printf("%d\n", cycles);

}
