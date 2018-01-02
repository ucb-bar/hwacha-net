#include <stdio.h>
#include "layer.h"
#include <math.h>
#include "imagenet_labels.h"
/* #define STB_IMAGE_IMPLEMENTATION */
/* #define STBI_ONLY_PNG */
/* #include "stb_image.h" */
#include "util.h"
#define NLAYERS 90

void swap(float** a, float** b)
{
  float* t = *a;
  *a = *b;
  *b = t;
}

int main(int argc, char** argv)
{
  hwacha_init();
  if (argc < 2)
    {
      printf("Pass path to image\n");
      return 0;
    }

  layer \
    conv1, bias1, relu1, pool1,
    fire2_conv1, fire2_bias1, fire2_relu1,
    fire2_conv2, fire2_bias2, fire2_relu2,
    fire2_conv3, fire2_bias3, fire2_relu3,
    fire2_concat,
    fire3_conv1, fire3_bias1, fire3_relu1,
    fire3_conv2, fire3_bias2, fire3_relu2,
    fire3_conv3, fire3_bias3, fire3_relu3,
    fire3_concat,
    fire4_conv1, fire4_bias1, fire4_relu1,
    fire4_conv2, fire4_bias2, fire4_relu2,
    fire4_conv3, fire4_bias3, fire4_relu3,
    fire4_concat,
    pool4,
    fire5_conv1, fire5_bias1, fire5_relu1,
    fire5_conv2, fire5_bias2, fire5_relu2,
    fire5_conv3, fire5_bias3, fire5_relu3,
    fire5_concat,
    fire6_conv1, fire6_bias1, fire6_relu1,
    fire6_conv2, fire6_bias2, fire6_relu2,
    fire6_conv3, fire6_bias3, fire6_relu3,
    fire6_concat,
    fire7_conv1, fire7_bias1, fire7_relu1,
    fire7_conv2, fire7_bias2, fire7_relu2,
    fire7_conv3, fire7_bias3, fire7_relu3,
    fire7_concat,
    fire8_conv1, fire8_bias1, fire8_relu1,
    fire8_conv2, fire8_bias2, fire8_relu2,
    fire8_conv3, fire8_bias3, fire8_relu3,
    fire8_concat,
    pool8,
    fire9_conv1, fire9_bias1, fire9_relu1,
    fire9_conv2, fire9_bias2, fire9_relu2,
    fire9_conv3, fire9_bias3, fire9_relu3,
    fire9_concat,
    convfin, biasfin, relufin, poolfin;

  layer* layers[NLAYERS] = {
    &conv1, &bias1, &relu1, &pool1,
    &fire2_conv1, &fire2_bias1, &fire2_relu1,
    &fire2_conv2, &fire2_bias2, &fire2_relu2,
    &fire2_conv3, &fire2_bias3, &fire2_relu3,
    &fire2_concat,
    &fire3_conv1, &fire3_bias1, &fire3_relu1,
    &fire3_conv2, &fire3_bias2, &fire3_relu2,
    &fire3_conv3, &fire3_bias3, &fire3_relu3,
    &fire3_concat,
    &fire4_conv1, &fire4_bias1, &fire4_relu1,
    &fire4_conv2, &fire4_bias2, &fire4_relu2,
    &fire4_conv3, &fire4_bias3, &fire4_relu3,
    &fire4_concat,
    &pool4,
    &fire5_conv1, &fire5_bias1, &fire5_relu1,
    &fire5_conv2, &fire5_bias2, &fire5_relu2,
    &fire5_conv3, &fire5_bias3, &fire5_relu3,
    &fire5_concat,
    &fire6_conv1, &fire6_bias1, &fire6_relu1,
    &fire6_conv2, &fire6_bias2, &fire6_relu2,
    &fire6_conv3, &fire6_bias3, &fire6_relu3,
    &fire6_concat,
    &fire7_conv1, &fire7_bias1, &fire7_relu1,
    &fire7_conv2, &fire7_bias2, &fire7_relu2,
    &fire7_conv3, &fire7_bias3, &fire7_relu3,
    &fire7_concat,
    &fire8_conv1, &fire8_bias1, &fire8_relu1,
    &fire8_conv2, &fire8_bias2, &fire8_relu2,
    &fire8_conv3, &fire8_bias3, &fire8_relu3,
    &fire8_concat,
    &pool8,
    &fire9_conv1, &fire9_bias1, &fire9_relu1,
    &fire9_conv2, &fire9_bias2, &fire9_relu2,
    &fire9_conv3, &fire9_bias3, &fire9_relu3,
    &fire9_concat,
    &convfin, &biasfin, &relufin, &poolfin};

  float* input;
  float* output;
  float* workspace;

  layer start;

  {
    conv1.type \
      = fire2_conv1.type = fire2_conv2.type = fire2_conv3.type  \
      = fire3_conv1.type = fire3_conv2.type = fire3_conv3.type  \
      = fire4_conv1.type = fire4_conv2.type = fire4_conv3.type  \
      = fire5_conv1.type = fire5_conv2.type = fire5_conv3.type  \
      = fire6_conv1.type = fire6_conv2.type = fire6_conv3.type  \
      = fire7_conv1.type = fire7_conv2.type = fire7_conv3.type  \
      = fire8_conv1.type = fire8_conv2.type = fire8_conv3.type  \
      = fire9_conv1.type = fire9_conv2.type = fire9_conv3.type  \
      = convfin.type = CONVOLUTIONAL_ENCODED;
    bias1.type                                                  \
      = fire2_bias1.type = fire2_bias2.type = fire2_bias3.type  \
      = fire3_bias1.type = fire3_bias2.type = fire3_bias3.type  \
      = fire4_bias1.type = fire4_bias2.type = fire4_bias3.type  \
      = fire5_bias1.type = fire5_bias2.type = fire5_bias3.type  \
      = fire6_bias1.type = fire6_bias2.type = fire6_bias3.type  \
      = fire7_bias1.type = fire7_bias2.type = fire7_bias3.type  \
      = fire8_bias1.type = fire8_bias2.type = fire8_bias3.type  \
      = fire9_bias1.type = fire9_bias2.type = fire9_bias3.type  \
      = biasfin.type = BIAS;
    relu1.type                                                  \
      = fire2_relu1.type = fire2_relu2.type = fire2_relu3.type  \
      = fire3_relu1.type = fire3_relu2.type = fire3_relu3.type  \
      = fire4_relu1.type = fire4_relu2.type = fire4_relu3.type  \
      = fire5_relu1.type = fire5_relu2.type = fire5_relu3.type  \
      = fire6_relu1.type = fire6_relu2.type = fire6_relu3.type  \
      = fire7_relu1.type = fire7_relu2.type = fire7_relu3.type  \
      = fire8_relu1.type = fire8_relu2.type = fire8_relu3.type  \
      = fire9_relu1.type = fire9_relu2.type = fire9_relu3.type  \
      = relufin.type = RELU;
    fire2_concat.type = fire3_concat.type = fire4_concat.type \
      = fire5_concat.type = fire6_concat.type = fire7_concat.type \
      = fire8_concat.type = fire9_concat.type = CONCAT;
    pool1.type = pool4.type = pool8.type = MAXPOOL;
    poolfin.type = AVERAGE;

    conv1.n = 96; conv1.size = 7; conv1.stride = 2; conv1.pad = 0;
    pool1.n =  3; pool1.size = 3; pool1.stride = 2; pool1.pad = 0;
    pool4.n =  3; pool4.size = 3; pool4.stride = 2; pool4.pad = 0;
    pool8.n =  3; pool8.size = 3; pool8.stride = 2; pool8.pad = 0;

    fire2_conv1.n = fire3_conv1.n = 16;
    fire4_conv1.n = fire5_conv1.n = 32;
    fire6_conv1.n = fire7_conv1.n = 48;
    fire8_conv1.n = fire9_conv1.n = 64;
    fire2_conv1.size = fire3_conv1.size = fire4_conv1.size = fire5_conv1.size \
      = fire6_conv1.size = fire7_conv1.size = fire8_conv1.size = fire9_conv1.size = 1;
    fire2_conv1.stride = fire3_conv1.stride = fire4_conv1.stride = fire5_conv1.stride \
      = fire6_conv1.stride = fire7_conv1.stride = fire8_conv1.stride = fire9_conv1.stride = 1;
    fire2_conv1.pad = fire3_conv1.pad = fire4_conv1.pad = fire5_conv1.pad \
      = fire6_conv1.pad = fire7_conv1.pad = fire8_conv1.pad = fire9_conv1.pad = 0;

    fire2_conv2.n = fire3_conv2.n = 64;
    fire4_conv2.n = fire5_conv2.n = 128;
    fire6_conv2.n = fire7_conv2.n = 192;
    fire8_conv2.n = fire9_conv2.n = 256;
    fire2_conv2.size = fire3_conv2.size = fire4_conv2.size = fire5_conv2.size \
      = fire6_conv2.size = fire7_conv2.size = fire8_conv2.size = fire9_conv2.size = 1;
    fire2_conv2.stride = fire3_conv2.stride = fire4_conv2.stride = fire5_conv2.stride \
      = fire6_conv2.stride = fire7_conv2.stride = fire8_conv2.stride = fire9_conv2.stride = 1;
    fire2_conv2.pad = fire3_conv2.pad = fire4_conv2.pad = fire5_conv2.pad \
      = fire6_conv2.pad = fire7_conv2.pad = fire8_conv2.pad = fire9_conv2.pad = 0;

    fire2_conv3.n = fire3_conv3.n = 64;
    fire4_conv3.n = fire5_conv3.n = 128;
    fire6_conv3.n = fire7_conv3.n = 192;
    fire8_conv3.n = fire9_conv3.n = 256;
    fire2_conv3.size = fire3_conv3.size = fire4_conv3.size = fire5_conv3.size \
      = fire6_conv3.size = fire7_conv3.size = fire8_conv3.size = fire9_conv3.size = 3;
    fire2_conv3.stride = fire3_conv3.stride = fire4_conv3.stride = fire5_conv3.stride \
      = fire6_conv3.stride = fire7_conv3.stride = fire8_conv3.stride = fire9_conv3.stride = 1;
    fire2_conv3.pad = fire3_conv3.pad = fire4_conv3.pad = fire5_conv3.pad \
      = fire6_conv3.pad = fire7_conv3.pad = fire8_conv3.pad = fire9_conv3.pad = 1;

    convfin.n = 1000; convfin.size = 1; convfin.stride = 1; convfin.pad = 1;

    start.type = START;
    start.output_h = start.output_w = 227;
    start.output_c = 3;
    start.prec = SINGLE;
  }

  {
    setup_layers(&start, &conv1);
    setup_layers(&conv1, &bias1);
    setup_layers(&bias1, &relu1);
    setup_layers(&relu1, &pool1);
    
    setup_layers(&pool1, &fire2_conv1);
    setup_layers(&fire2_conv1, &fire2_bias1);
    setup_layers(&fire2_bias1, &fire2_relu1);

    setup_layers(&fire2_relu1, &fire2_conv2);
    setup_layers(&fire2_conv2, &fire2_bias2);
    setup_layers(&fire2_bias2, &fire2_relu2);

    setup_layers(&fire2_relu1, &fire2_conv3);
    setup_layers(&fire2_conv3, &fire2_bias3);
    setup_layers(&fire2_bias3, &fire2_relu3);

    concat_layers(&fire2_relu2, &fire2_relu3, &fire2_concat);

    setup_layers(&fire2_concat, &fire3_conv1);
    setup_layers(&fire3_conv1, &fire3_bias1);
    setup_layers(&fire3_bias1, &fire3_relu1);

    setup_layers(&fire3_relu1, &fire3_conv2);
    setup_layers(&fire3_conv2, &fire3_bias2);
    setup_layers(&fire3_bias2, &fire3_relu2);

    setup_layers(&fire3_relu1, &fire3_conv3);
    setup_layers(&fire3_conv3, &fire3_bias3);
    setup_layers(&fire3_bias3, &fire3_relu3);

    concat_layers(&fire3_relu2, &fire3_relu3, &fire3_concat);

    setup_layers(&fire3_concat, &fire4_conv1);
    setup_layers(&fire4_conv1, &fire4_bias1);
    setup_layers(&fire4_bias1, &fire4_relu1);

    setup_layers(&fire4_relu1, &fire4_conv2);
    setup_layers(&fire4_conv2, &fire4_bias2);
    setup_layers(&fire4_bias2, &fire4_relu2);

    setup_layers(&fire4_relu1, &fire4_conv3);
    setup_layers(&fire4_conv3, &fire4_bias3);
    setup_layers(&fire4_bias3, &fire4_relu3);

    concat_layers(&fire4_relu2, &fire4_relu3, &fire4_concat);

    setup_layers(&fire4_concat, &pool4);
    
    setup_layers(&pool4, &fire5_conv1);
    setup_layers(&fire5_conv1, &fire5_bias1);
    setup_layers(&fire5_bias1, &fire5_relu1);

    setup_layers(&fire5_relu1, &fire5_conv2);
    setup_layers(&fire5_conv2, &fire5_bias2);
    setup_layers(&fire5_bias2, &fire5_relu2);

    setup_layers(&fire5_relu1, &fire5_conv3);
    setup_layers(&fire5_conv3, &fire5_bias3);
    setup_layers(&fire5_bias3, &fire5_relu3);

    concat_layers(&fire5_relu2, &fire5_relu3, &fire5_concat);

    
    setup_layers(&fire5_concat, &fire6_conv1);
    setup_layers(&fire6_conv1, &fire6_bias1);
    setup_layers(&fire6_bias1, &fire6_relu1);

    setup_layers(&fire6_relu1, &fire6_conv2);
    setup_layers(&fire6_conv2, &fire6_bias2);
    setup_layers(&fire6_bias2, &fire6_relu2);

    setup_layers(&fire6_relu1, &fire6_conv3);
    setup_layers(&fire6_conv3, &fire6_bias3);
    setup_layers(&fire6_bias3, &fire6_relu3);

    concat_layers(&fire6_relu2, &fire6_relu3, &fire6_concat);

    
    setup_layers(&fire6_concat, &fire7_conv1);
    setup_layers(&fire7_conv1, &fire7_bias1);
    setup_layers(&fire7_bias1, &fire7_relu1);

    setup_layers(&fire7_relu1, &fire7_conv2);
    setup_layers(&fire7_conv2, &fire7_bias2);
    setup_layers(&fire7_bias2, &fire7_relu2);

    setup_layers(&fire7_relu1, &fire7_conv3);
    setup_layers(&fire7_conv3, &fire7_bias3);
    setup_layers(&fire7_bias3, &fire7_relu3);

    concat_layers(&fire7_relu2, &fire7_relu3, &fire7_concat);

    
    setup_layers(&fire7_concat, &fire8_conv1);
    setup_layers(&fire8_conv1, &fire8_bias1);
    setup_layers(&fire8_bias1, &fire8_relu1);

    setup_layers(&fire8_relu1, &fire8_conv2);
    setup_layers(&fire8_conv2, &fire8_bias2);
    setup_layers(&fire8_bias2, &fire8_relu2);

    setup_layers(&fire8_relu1, &fire8_conv3);
    setup_layers(&fire8_conv3, &fire8_bias3);
    setup_layers(&fire8_bias3, &fire8_relu3);

    concat_layers(&fire8_relu2, &fire8_relu3, &fire8_concat);

    setup_layers(&fire8_concat, &pool8);
      
    setup_layers(&pool8, &fire9_conv1);
    setup_layers(&fire9_conv1, &fire9_bias1);
    setup_layers(&fire9_bias1, &fire9_relu1);

    setup_layers(&fire9_relu1, &fire9_conv2);
    setup_layers(&fire9_conv2, &fire9_bias2);
    setup_layers(&fire9_bias2, &fire9_relu2);

    setup_layers(&fire9_relu1, &fire9_conv3);
    setup_layers(&fire9_conv3, &fire9_bias3);
    setup_layers(&fire9_bias3, &fire9_relu3);

    concat_layers(&fire9_relu2, &fire9_relu3, &fire9_concat);

    setup_layers(&fire9_concat, &convfin);
    setup_layers(&convfin, &biasfin);
    setup_layers(&biasfin, &relufin);
    setup_layers(&relufin, &poolfin);

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
    FILE* fp = fopen("weights/squeezenet_encoded_single.weights", "rb");
    load_layers(layers, NLAYERS, fp);
    fclose(fp);
  }
  {

    FILE* fp = fopen(argv[1], "rb");
    fread(input, sizeof(float), 227*227*3, fp);
    fclose(fp);
    
    printf("%.3f input\n", input[0]);
  }
  {
    layer_forward(&conv1, input, output, workspace); swap(&input, &output);
    layer_forward(&bias1, input, output, workspace);
    layer_forward(&relu1, input, output, workspace);
    layer_forward(&pool1, input, output, workspace);
    

    layer_forward(&fire2_conv1, input, output, workspace); swap(&input, &output);
    layer_forward(&fire2_bias1, input, output, workspace);
    layer_forward(&fire2_relu1, input, output, workspace);

    layer_forward(&fire2_conv2, input, output, workspace);
    layer_forward(&fire2_conv3, input, output + fire2_conv2.output_size, workspace);
    swap(&input, &output);
    layer_forward(&fire2_bias2, input, output, workspace);
    layer_forward(&fire2_relu2, input, output, workspace);
    layer_forward(&fire2_bias3, input + fire2_conv2.output_size, output, workspace);
    layer_forward(&fire2_relu3, input + fire2_conv2.output_size, output, workspace);

    
    layer_forward(&fire3_conv1, input, output, workspace); swap(&input, &output);
    layer_forward(&fire3_bias1, input, output, workspace);
    layer_forward(&fire3_relu1, input, output, workspace);

    layer_forward(&fire3_conv2, input, output, workspace);
    layer_forward(&fire3_conv3, input, output + fire3_conv2.output_size, workspace);
    swap(&input, &output);
    layer_forward(&fire3_bias2, input, output, workspace);
    layer_forward(&fire3_relu2, input, output, workspace);
    layer_forward(&fire3_bias3, input + fire3_conv2.output_size, output, workspace);
    layer_forward(&fire3_relu3, input + fire3_conv2.output_size, output, workspace);
    

    layer_forward(&fire4_conv1, input, output, workspace); swap(&input, &output);
    layer_forward(&fire4_bias1, input, output, workspace);
    layer_forward(&fire4_relu1, input, output, workspace);

    layer_forward(&fire4_conv2, input, output, workspace);
    layer_forward(&fire4_conv3, input, output + fire4_conv2.output_size, workspace);
    swap(&input, &output);
    layer_forward(&fire4_bias2, input, output, workspace);
    layer_forward(&fire4_relu2, input, output, workspace);
    layer_forward(&fire4_bias3, input + fire4_conv2.output_size, output, workspace);
    layer_forward(&fire4_relu3, input + fire4_conv2.output_size, output, workspace);

    layer_forward(&pool4, input, output, workspace);

    layer_forward(&fire5_conv1, input, output, workspace); swap(&input, &output);
    layer_forward(&fire5_bias1, input, output, workspace);
    layer_forward(&fire5_relu1, input, output, workspace);

    layer_forward(&fire5_conv2, input, output, workspace);
    layer_forward(&fire5_conv3, input, output + fire5_conv2.output_size, workspace);
    swap(&input, &output);
    layer_forward(&fire5_bias2, input, output, workspace);
    layer_forward(&fire5_relu2, input, output, workspace);
    layer_forward(&fire5_bias3, input + fire5_conv2.output_size, output, workspace);
    layer_forward(&fire5_relu3, input + fire5_conv2.output_size, output, workspace);
    

    layer_forward(&fire6_conv1, input, output, workspace); swap(&input, &output);
    layer_forward(&fire6_bias1, input, output, workspace);
    layer_forward(&fire6_relu1, input, output, workspace);

    layer_forward(&fire6_conv2, input, output, workspace);
    layer_forward(&fire6_conv3, input, output + fire6_conv2.output_size, workspace);
    swap(&input, &output);
    layer_forward(&fire6_bias2, input, output, workspace);
    layer_forward(&fire6_relu2, input, output, workspace);
    layer_forward(&fire6_bias3, input + fire6_conv2.output_size, output, workspace);
    layer_forward(&fire6_relu3, input + fire6_conv2.output_size, output, workspace);
    

    layer_forward(&fire7_conv1, input, output, workspace); swap(&input, &output);
    layer_forward(&fire7_bias1, input, output, workspace);
    layer_forward(&fire7_relu1, input, output, workspace);

    layer_forward(&fire7_conv2, input, output, workspace);
    layer_forward(&fire7_conv3, input, output + fire7_conv2.output_size, workspace);
    swap(&input, &output);
    layer_forward(&fire7_bias2, input, output, workspace);
    layer_forward(&fire7_relu2, input, output, workspace);
    layer_forward(&fire7_bias3, input + fire7_conv2.output_size, output, workspace);
    layer_forward(&fire7_relu3, input + fire7_conv2.output_size, output, workspace);


    layer_forward(&fire8_conv1, input, output, workspace); swap(&input, &output);
    layer_forward(&fire8_bias1, input, output, workspace);
    layer_forward(&fire8_relu1, input, output, workspace);

    layer_forward(&fire8_conv2, input, output, workspace);
    layer_forward(&fire8_conv3, input, output + fire8_conv2.output_size, workspace);
    swap(&input, &output);
    layer_forward(&fire8_bias2, input, output, workspace);
    layer_forward(&fire8_relu2, input, output, workspace);
    layer_forward(&fire8_bias3, input + fire8_conv2.output_size, output, workspace);
    layer_forward(&fire8_relu3, input + fire8_conv2.output_size, output, workspace);

    layer_forward(&pool8, input, output, workspace);

    layer_forward(&fire9_conv1, input, output, workspace); swap(&input, &output);
    layer_forward(&fire9_bias1, input, output, workspace);
    layer_forward(&fire9_relu1, input, output, workspace);

    layer_forward(&fire9_conv2, input, output, workspace);
    layer_forward(&fire9_conv3, input, output + fire9_conv2.output_size, workspace);
    swap(&input, &output);
    layer_forward(&fire9_bias2, input, output, workspace);
    layer_forward(&fire9_relu2, input, output, workspace);
    layer_forward(&fire9_bias3, input + fire9_conv2.output_size, output, workspace);
    layer_forward(&fire9_relu3, input + fire9_conv2.output_size, output, workspace);

    layer_forward(&convfin, input, output, workspace); swap(&input, &output);
    layer_forward(&biasfin, input, output, workspace);
    layer_forward(&relufin, input, output, workspace);
    layer_forward(&poolfin, input, output, workspace);
  }
  {
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
