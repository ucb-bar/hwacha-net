#include "fc_layer.h"
#include "util.h"

void fc_forward_32(struct layer* l, float* src, float* dest, float* workspace)
{
  fill_32(l->output_h*l->output_w*l->output_c, 0.0, dest);

  float* weights = l->weights_32;
  for (int i = 0; i < l->w*l->h*l->c; i++)
    {
      if (src[i] != 0.0)
        axpy_32(l->output_h*l->output_w*l->output_c, src[i], &weights[i*l->output_c], dest);
    }
  printf("%.8f %.8f fc\n", dest[0], dest[999]);
}
