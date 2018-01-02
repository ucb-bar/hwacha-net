#ifndef layer_h
#define layer_h

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>


typedef enum {
  DOUBLE = 8,
  SINGLE = 4,
  HALF = 2
} PRECISION;
typedef enum {
  ERROR,
  CONVOLUTIONAL,
  CONVOLUTIONAL_ENCODED,
  CONVOLUTIONAL_ENCODED_COMPRESSED,
  MAXPOOL,
  MAXPOOL_DARKNET,
  BATCHNORM,
  REGION,
  START,
  BIAS,
  RELU,
  LEAKY,
  CONCAT,
  AVERAGE
} LAYER_TYPE;

struct layer;
typedef struct layer layer;

struct layer
{
  PRECISION prec;
  LAYER_TYPE type;
  int h;
  int w;
  int c;


  int stride;
  int size;
  int n;
  int pad;

  int nids;
  int nindptr;
  int ndata;

  int16_t* weights_16;
  float* weights_32;
  int* indices;

  int8_t* encoded_indices;
  int8_t* encoded_indptr;
  int8_t* encoded_data;

  size_t input_size;
  size_t workspace_size;
  size_t output_size;

  int output_h;
  int output_w;
  int output_c;

  int nweights;
};

void setup_layers(struct layer*, struct layer*);
void concat_layers(struct layer*, struct layer*, struct layer*);
size_t max_size(struct layer**, int);
size_t max_workspace(struct layer**, int);
void load_layers(struct layer**, int, FILE*);

void layer_forward(struct layer*, void*, void*, void*);
#endif
