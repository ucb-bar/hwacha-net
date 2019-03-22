#include <stdio.h>
#include "layer.h"
#include <math.h>
#include "imagenet_labels.h"
/* #define STB_IMAGE_IMPLEMENTATION */
/* #define STBI_ONLY_PNG */
/* #include "stb_image.h" */
#include "util.h"
#include "parse_args.h"
#define NLAYERS 29

void swap(float** a, float** b)
{
  float* t = *a;
  *a = *b;
  *b = t;
}

int main(int argc, char** argv)
{
  char* images[100] = {NULL}; // Do not give me more than 100 images!
  char weights[128] = "weights/alexnet_encoded_single.weights";

  if (!parse_args(argc, argv, weights, images)) {
    fprintf(stderr, "No image is given!\n");
    return -2;
  }

  hwacha_init();

  layer \
    conv1, bias1, relu1, norm1, pool1,
    conv2, bias2, relu2, norm2, pool2,
    conv3, bias3, relu3,
    conv4, bias4, relu4,
    conv5, bias5, relu5, pool5,
    fc6, bias6, relu6,
    fc7, bias7, relu7,
    fc8, bias8, softmax;

  layer* layers[NLAYERS] = {
    &conv1, &bias1, &relu1, &norm1, &pool1,
    &conv2, &bias2, &relu2, &norm2, &pool2,
    &conv3, &bias3, &relu3,
    &conv4, &bias4, &relu4,
    &conv5, &bias5, &relu5, &pool5,
    &fc6, &bias6, &relu6,
    &fc7, &bias7, &relu7,
    &fc8, &bias8, &softmax };
  float* input;
  float* output;
  float* workspace;

  layer start;

  {
    conv1.type = conv3.type = CONVOLUTIONAL_ENCODED;
    conv2.type = conv4.type = conv5.type = CONVOLUTIONAL_GROUPED_ENCODED;
    bias1.type = bias2.type = bias3.type = bias4.type = bias5.type \
      = bias6.type = bias7.type = bias8.type = BIAS;
    relu1.type = relu2.type = relu3.type = relu4.type = relu5.type \
      = relu6.type = relu7.type = RELU;
    norm1.type = norm2.type = NORMALIZATION;
    pool1.type = pool2.type = pool5.type = MAXPOOL;
    fc6.type = fc7.type = fc8.type = FULLY_CONNECTED_ENCODED;

    conv1.n = 96; conv1.size = 11; conv1.stride = 4; conv1.pad = 0;
    norm1.size = 5; norm1.alpha = 0.0001; norm1.beta = 0.75;
    pool1.n =  3; pool1.size = 3; pool1.stride = 2; pool1.pad = 0;

    conv2.n = 256; conv2.size = 5; conv2.stride = 1; conv2.pad = 2; conv2.groups = 2;
    norm2.size = 5; norm2.alpha = 0.0001; norm2.beta = 0.75;
    pool2.n = 3; pool2.size = 3; pool2.stride = 2; pool2.pad = 0;

    conv3.n = 384; conv3.size = 3; conv3.stride = 1; conv3.pad = 1;
    conv4.n = 384; conv4.size = 3; conv4.stride = 1; conv4.pad = 1; conv4.groups = 2;
    conv5.n = 256; conv5.size = 3; conv5.stride = 1; conv5.pad = 1; conv5.groups = 2;

    pool5.n = 3; pool5.size = 3; pool5.stride = 2; pool5.pad = 0;

    fc6.n = 4096;
    fc7.n = 4096;
    fc8.n = 1000;

    softmax.type = SOFTMAX;

    start.type = START;
    start.output_h = start.output_w = 227;
    start.output_c = 3;
    start.prec = SINGLE;
  }

  {
    setup_layers(&start, &conv1);
    for (int i = 1; i < NLAYERS; i++)
      setup_layers(layers[i-1], layers[i]);
  }

  {
    size_t max_io = max_size(layers, NLAYERS);
    size_t max_ws = max_workspace(layers, NLAYERS);
    //printf("%lu %lu\n", max_io, max_ws);
    input = safe_malloc(sizeof(float)*max_io);
    output = safe_malloc(sizeof(float)*max_io);
    workspace = safe_malloc(sizeof(float)*max_ws);
  }
  {
    FILE* fp = fopen(weights, "rb");
    printf("weights: %s\n", weights);
    load_layers(layers, NLAYERS, fp);
    fclose(fp);
  }

  for (int image_id = 0 ; images[image_id] ; image_id++)
  {
    char* image = images[image_id];
    printf("image: %s\n", image);
    FILE* fp = fopen(image, "rb");
    if (!fp) {
      perror(image);
      return -2;
    }
    fread(input, sizeof(float), 227*227*3, fp);
    fclose(fp);
    free(image);
    printf("%.3f\n", input[0]);

    layer_forward(&conv1, input, output, workspace); swap(&input, &output);
    layer_forward(&bias1, input, output, workspace);
    layer_forward(&relu1, input, output, workspace);
    layer_forward(&norm1, input, output, workspace); swap(&input, &output);
    layer_forward(&pool1, input, output, workspace);

    layer_forward(&conv2, input, output, workspace); swap(&input, &output);
    layer_forward(&bias2, input, output, workspace);
    layer_forward(&relu2, input, output, workspace);
    layer_forward(&norm2, input, output, workspace); swap(&input, &output);
    layer_forward(&pool2, input, output, workspace);

    layer_forward(&conv3, input, output, workspace); swap(&input, &output);
    layer_forward(&bias3, input, output, workspace);
    layer_forward(&relu3, input, output, workspace);

    layer_forward(&conv4, input, output, workspace); swap(&input, &output);
    layer_forward(&bias4, input, output, workspace);
    layer_forward(&relu4, input, output, workspace);

    layer_forward(&conv5, input, output, workspace); swap(&input, &output);
    layer_forward(&bias5, input, output, workspace);
    layer_forward(&relu5, input, output, workspace);
    layer_forward(&pool5, input, output, workspace);

    layer_forward(&fc6, input, output, workspace); swap(&input, &output);
    layer_forward(&bias6, input, output, workspace);
    layer_forward(&relu6, input, output, workspace);

    layer_forward(&fc7, input, output, workspace); swap(&input, &output);
    layer_forward(&bias7, input, output, workspace);
    layer_forward(&relu7, input, output, workspace);

    layer_forward(&fc8, input, output, workspace); swap(&input, &output);
    layer_forward(&bias8, input, output, workspace);
    layer_forward(&softmax, input, output, workspace); swap(&input, &output);

    float max = 0;
    int maxi = -1;
    for (int i = 0; i < 1000; i++)
      {
        if (input[i] > max)
          {
            max = input[i];
            maxi = i;
          }
      }
    printf("Detected %s\n", LABELS[maxi]);
  }
  return 0;
}
