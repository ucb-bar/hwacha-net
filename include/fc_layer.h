#ifndef FC_LAYER_H
#define FC_LAYER_H

#include "layer.h"

void fc_forward_32(struct layer* l, float* src, float* dest, float* workspace);
void fc_forward_encoded_32(struct layer* l, float* src, float* dest, float* workspace);
#endif
