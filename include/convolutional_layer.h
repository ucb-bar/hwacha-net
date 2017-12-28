#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "layer.h"

int conv_out_width(struct layer *l);
int conv_out_height(struct layer *l);

void im2col_id(struct layer* l);
void convolutional_precomp_forward_16(struct layer* l, int16_t* src, int16_t* dest, int16_t* workspace);
void convolutional_precomp_forward_32(struct layer* l, float* src, float* dest, float* workspace);
void convolutional_forward_32(struct layer* l, float* src, float* dest, float* workspace);
#endif
