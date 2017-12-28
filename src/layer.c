#include "layer.h"
#include "util.h"
#include <math.h>
#include <stdint.h>
#include "convolutional_layer.h"
#include "maxpool_layer.h"

void setup_layers(struct layer* l1, struct layer* l2)
{
  l2->prec = l1->prec;
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
        l2->output_w = conv_out_width(l2);
        l2->output_h = conv_out_height(l2);
        l2->output_c = l2->n;
        l2->workspace_size = l2->output_h * l2->output_w * l2->c * l2->size * l2->size;
        l2->nweights = l2->size * l2->size * l2->c * l2->n;
        im2col_id(l2, l2->prec);
        break;
      }
    case MAXPOOL_DARKNET:
      {
        l2->output_w = (l2->w + 2*l2->pad)/l2->stride;
        l2->output_h = (l2->h + 2*l2->pad)/l2->stride;
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
        l2->output_w = conv_out_width(l2);
        l2->output_h = conv_out_height(l2);
        break;
      }
    case AVERAGE:
      {
        l2->output_w = l2->output_h = 1;
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

void concat_layers(struct layer* l1, struct layer* l2, struct layer* l3)
{
  if (l3->type != CONCAT)
    {
      printf("ERROR NOT CONCAT LAYER\n");
      return;
    }
  if (l1->h != l2->h ||
      l1->w != l2->w ||
      l1->prec != l2->prec)
    {
      printf("ERROR DIM NOT ALIGNED\n");
      return;
    }
  l3->prec = l1->prec;
  l3->h = l1->h;
  l3->w = l1->w;
  l3->c = -1;
  l3->workspace_size = 0;
  l3->nweights = 0;
  l3->output_w = l3->w;
  l3->output_h = l3->h;
  l3->output_c = l1->c + l2->c;
  l3->input_size = 0;
  l2->output_size = l2->output_w * l2->output_h * l2->output_c;
  //printf("%d %d %d + %d \t %d %d %d \n", l2->h, l2->w, l1->c, l2->c, l3->output_h, l3->output_w, l3->output_c);
  
}
void load_layers(struct layer** layers, int n, FILE* fp)
{
  for (int i = 0; i < n; i++) {
    layer* l = layers[i];
    switch (l->prec)
      {
      case DOUBLE : { break;}
      case SINGLE :
        {
          l->weights_32 = safe_malloc(l->nweights * sizeof(float));
          fread(l->weights_32, sizeof(float), l->nweights, fp);
          //if (l->nweights) printf("%.6f\n", l->weights_32[0]);
          break;
        }
      case HALF :
        {
          l->weights_16 = safe_malloc(l->nweights * sizeof(int16_t));
          fread(l->weights_16, sizeof(int16_t), l->nweights, fp);
          //if (l->nweights) printf("%hu\n", l->weights_16[0]);
          break;
        }
      }

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


void batchnorm_forward_16(struct layer* l, int16_t* src, int16_t* dest)
{
  int16_t* scales = l->weights_16;
  int16_t* rolling_mean = l->weights_16 + l->output_c;
  int16_t* rolling_variance = l->weights_16 + 2 * l->output_c;
  normalize_16(src, rolling_mean, rolling_variance, l->output_c, l->output_h*l->output_w);
  for(int i = 0; i < l->output_c; ++i)
    scale_16 (&src[i*l->output_h*l->output_w], scales[i], l->output_h*l->output_w);
}

void batchnorm_forward_32(struct layer* l, float* src, float* dest)
{
  float* scales = l->weights_32;
  float* rolling_mean = l->weights_32 + l->output_c;
  float* rolling_variance = l->weights_32 + 2 * l->output_c;
  normalize_32(src, rolling_mean, rolling_variance, l->output_c, l->output_h*l->output_w);
  for(int i = 0; i < l->output_c; ++i)
    scale_32 (&src[i*l->output_h*l->output_w], scales[i], l->output_h*l->output_w);
  //printf("%.3f %.3f %.3f batchnorm \n", src[0], src[1], src[2]);
}
void bias_forward_16(struct layer* l, int16_t* src)
{
  for(int i = 0; i < l->output_c; ++i)
    add_16 (&src[i*l->output_h*l->output_w], l->weights_16[i], l->output_h*l->output_w);
}
void bias_forward_32(struct layer* l, float* src)
{
  for(int i = 0; i < l->output_c; ++i)
    add_32 (&src[i*l->output_h*l->output_w], l->weights_32[i], l->output_h*l->output_w);
  //  printf("%.3f bias\n", src[0]);
}

void leaky_forward_16(struct layer* l, int16_t* src)
{
  setvcfg(0, 0, 1, 2);
  float a = 0.1f;
  asm volatile ("vmcs vs2, %0" : : "r" (a));
  int len = l->output_h*l->output_w*l->output_c;
  for (int i = 0; i < len; )
    {
      int consumed = setvlen(len - i);
      asm volatile ("vmca va0, %0" : : "r" (&src[i]));
      asm volatile ("la t0, vleaky_activate_16" : : : "t0");
      asm volatile ("lw t1, 0(t0)");
      asm volatile ("vf 0(t0)");
      i += consumed;
    }
  asm volatile ("fence");
}
void leaky_forward_32(struct layer* l, float* src)
{
  setvcfg(0, 1, 0, 2);
  float a = 0.1f;
  asm volatile ("vmcs vs2, %0" : : "r" (a));
  int len = l->output_h*l->output_w*l->output_c;
  for (int i = 0; i < len; )
    {
      int consumed = setvlen(len - i);
      asm volatile ("vmca va0, %0" : : "r" (&src[i]));
      asm volatile ("la t0, vleaky_activate_32" : : : "t0");
      asm volatile ("lw t1, 0(t0)");
      asm volatile ("vf 0(t0)");
      i += consumed;
    }
  asm volatile ("fence");
  //printf("%.3f %.3f %.3f leaky \n", src[0], src[1], src[2]);
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
void logistic_32(float* src, int size)
{
  for (int i = 0; i < size; i++)
    {
      src[i] = logistic_activate(src[i]);
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
void region_forward_16(struct layer* l, int16_t* src, int16_t* dest, int16_t* workspace)
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

void region_forward_32(struct layer* l, float* src, float* dest, float* workspace)
{
  int index;
  memcpy_32(src, dest, l->h*l->w*l->c);
  for(int n = 0; n < l->n; ++n)
    {
      index = entry_index(l, n*l->w*l->h, 0);
      logistic_32(dest+index, 2*l->w*l->h);
      index = entry_index(l, n*l->w*l->h, 4);
      logistic_32(dest+index, l->w*l->h);
    }
  index = entry_index(l, 0, 4 + 1);
  softmax_cpu(src + index, l->size, l->n, l->h*l->w*l->c/l->n, l->w*l->h, l->w*l->h, dest + index);
}
void relu_forward_32(struct layer* l, float* src)
{
  setvcfg(0, 1, 0, 1);
  int len = l->output_h*l->output_w*l->output_c;
  for (int i = 0; i < len; )
    {
      int consumed = setvlen(len - i);
      asm volatile ("vmca va0, %0" : : "r" (&src[i]));
      asm volatile ("la t0, vrelu_activate_32" : : : "t0");
      asm volatile ("lw t1, 0(t0)");
      asm volatile ("vf 0(t0)");
      i += consumed;
    }
  asm volatile ("fence");
  //printf("%.3f %.3f %.3f relu \n", src[0], src[1], src[2]);
}
void average_forward_32(struct layer* l, float* src)
{
  float count = l->h*l->w;
  float* srcptr = src;
  for (int k = 0; k < l->c; k++)
    {
      float acc = 0.0;
      for (int i = 0; i < l->h*l->w; i++)
        acc += srcptr[i];
      src[k] = acc / count;
      srcptr += l->h*l->w;
    }
  //printf("%.3f average\n", src[0]);
}
void layer_forward_16(struct layer* layer, int16_t* src, int16_t* dest, int16_t* workspace)
{
  switch (layer->type)
    {
    case CONVOLUTIONAL: { convolutional_precomp_forward_16(layer, src, dest, workspace); break; }
    case MAXPOOL_DARKNET: { maxpool_darknet_forward_16(layer, src, dest); break; }
    case BATCHNORM: { batchnorm_forward_16(layer, src, dest); break; }
    case REGION: { region_forward_16(layer, src, dest, workspace); break; }
    case BIAS: { bias_forward_16(layer, src); break; }
    case LEAKY: { leaky_forward_16(layer, src); break; }
    default: { printf("Unknown layer\n"); break; }
    }
}
void layer_forward_32(struct layer* layer, float* src, float* dest, float* workspace)
{
  switch (layer->type)
    {
    case CONVOLUTIONAL: { convolutional_precomp_forward_32(layer, src, dest, workspace); break; }
    case MAXPOOL_DARKNET: { maxpool_darknet_forward_32(layer, src, dest); break; }
    case BATCHNORM: { batchnorm_forward_32(layer, src, dest); break; }
    case REGION: { region_forward_32(layer, src, dest, workspace); break; }
    case BIAS: { bias_forward_32(layer, src); break; }
    case LEAKY: { leaky_forward_32(layer, src); break; }
    case RELU: { relu_forward_32(layer, src); break; }
    case MAXPOOL: {maxpool_darknet_forward_32(layer, src, dest); break;}
    case AVERAGE: {average_forward_32(layer, src); break;}
    default: { printf("Unknown layer\n"); break; }
    }
}
void layer_forward_64(struct layer* layer, double* src, double* dest, double* workspace)
{

}

void layer_forward(struct layer* layer, void* src, void* dest, void* workspace)
{
  size_t cycles = rdcycle();
  switch (layer->prec)
    {
    case HALF: {layer_forward_16(layer, src, dest, workspace); break;}
    case SINGLE: {layer_forward_32(layer, src, dest, workspace); break;}
    case DOUBLE: {layer_forward_64(layer, src, dest, workspace); break;}
    }
  cycles = rdcycle() - cycles;
  //printf("%lu\n", cycles);
}
