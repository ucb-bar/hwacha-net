#include <stdio.h>
#include "layer.h"
#include "util.h"

int main (int argc, char** argv)
{
  printf("TEST EXEC\n");

  layer conv1;
  layer* layers[1] = {&conv1};
  float* input;
  float* output;
  float* workspace;

  layer start;

  conv1.type = CONVOLUTIONAL_ENCODED;
  conv1.n = 16; conv1.size = 11; conv1.stride = 2; conv1.pad = 0;

  start.type = START;
  start.output_h = start.output_w = 227;
  start.output_c = 3;
  start.prec = SINGLE;

  setup_layers (&start, &conv1);

  printf("%d \n", conv1.output_w);
  return 0;
  size_t max_io = max_size(layers, 1);
  size_t max_ws = max_workspace(layers, 1);

  input = safe_malloc(sizeof(float)*max_io);
  output = safe_malloc(sizeof(float)*max_io);
  workspace = safe_malloc(sizeof(float)*max_ws);

  FILE* fp = fopen("weights/squeezenet_encoded_single.weights", "rb");
  load_layers(layers, 1, fp);
  fclose(fp);

  fill_seq_32(input, 15*15*3, 1);

  printf("%.3f input\n", input[0]);
  size_t cycles = rdcycle();
  layer_forward(&conv1, input, output, workspace);
  cycles = rdcycle() - cycles;
  printf("%lu cycles\n", cycles);
  return 0;
}
