#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H
#include "layer.h"

void maxpool_darknet_id(struct layer* l);
void maxpool_darknet_precomp_forward(struct layer* l, int16_t* src, int16_t* dest);
void maxpool_darknet_forward_16(struct layer* l, int16_t* src, int16_t* dest);
void maxpool_darknet_forward_32(struct layer* l, float* src, float* dest);
#endif
