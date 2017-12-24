#ifndef layer_h
#define layer_h

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>


typedef enum {
  ERROR,
  CONVOLUTIONAL,
  MAXPOOL,
  MAXPOOL_DARKNET,
  BATCHNORM,
  REGION,
  START,
  BIAS,
  RELU,
  LEAKY,
} LAYER_TYPE;

struct layer;
typedef struct layer layer;

struct layer
{
  LAYER_TYPE type;
  int h;
  int w;
  int c;
  
  
  int stride;
  int size;
  int n;
  int pad;

  int16_t* weights_16;
  int* indices;

  size_t input_size;
  size_t workspace_size;
  size_t output_size;

  int output_h;
  int output_w;
  int output_c;

  int nweights;
};

void setup_layers(struct layer*, struct layer*);
size_t max_size(struct layer**, int);
size_t max_workspace(struct layer**, int);
void load_layers(struct layer**, int, FILE*);

void layer_forward(struct layer*, int16_t*, int16_t*, int16_t*);
#endif
