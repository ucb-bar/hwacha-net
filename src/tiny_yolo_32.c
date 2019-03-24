#include <stdio.h>
#include "layer.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "util.h"
#include "parse_args.h"

#define NLAYERS 41

char* labels[20] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
                    "train", "tvmonitor" };
void swap(float** a, float** b)
{
  void* t = *a;
  *a = *b;
  *b = t;
}

int main(int argc, char** argv)
{
  char* images[100] = {NULL}; // Do not give me more than 100 images!
  char weights[128] = "weights/tiny_yolo_single.weights";

  if (!parse_args(argc, argv, weights, images)) {
    fprintf(stderr, "No image is given!\n");
    return -2;
  }

  hwacha_init();

  layer \
    conv0, bn0, bias0, leaky0, max1,
    conv2, bn2, bias2, leaky2, max3,
    conv4, bn4, bias4, leaky4, max5,
    conv6, bn6, bias6, leaky6, max7,
    conv8, bn8, bias8, leaky8, max9,
    conv10, bn10, bias10, leaky10, max11,
    conv12, bn12, bias12, leaky12,
    conv13, bn13, bias13, leaky13,
    conv14, bias14, region15;
  layer* layers[NLAYERS] = {
    &conv0, &bn0, &bias0, &leaky0, &max1,
    &conv2, &bn2, &bias2, &leaky2, &max3,
    &conv4, &bn4, &bias4, &leaky4, &max5,
    &conv6, &bn6, &bias6, &leaky6, &max7,
    &conv8, &bn8, &bias8, &leaky8, &max9,
    &conv10, &bn10, &bias10, &leaky10, &max11,
    &conv12, &bn12, &bias12, &leaky12,
    &conv13, &bn13, &bias13, &leaky13,
    &conv14, &bias14, &region15};

  float* input;
  float* output;
  float* workspace;

  // Create layers for tiny_yolo
  {
    conv0.type = conv2.type = conv4.type = conv6.type = conv8.type = conv10.type = conv12.type = \
      conv13.type = conv14.type = CONVOLUTIONAL;
    max1.type = max3.type = max5.type = max7.type = max9.type = max11.type = MAXPOOL_DARKNET;
    bn0.type = bn2.type = bn4.type = bn6.type = bn8.type = bn10.type = bn12.type = bn13.type = BATCHNORM;
    bias0.type = bias2.type = bias4.type = bias6.type = bias8.type = bias10.type = bias12.type \
      = bias13.type = bias14.type = BIAS;
    leaky0.type = leaky2.type = leaky4.type = leaky6.type = leaky8.type = leaky10.type = leaky12.type = leaky13.type = LEAKY;
    region15.type = REGION;
    
    conv0.n =    16; conv0.size =  3; conv0.stride =  1; conv0.pad =     1;
    max1.n =      2; max1.size =   2; max1.stride =   2; max1.pad =      0;
    conv2.n =    32; conv2.size =  3; conv2.stride =  1; conv2.pad =     1;
    max3.n =      2; max3.size =   2; max3.stride =   2; max3.pad =      0;
    conv4.n =    64; conv4.size =  3; conv4.stride =  1; conv4.pad =     1;
    max5.n =      2; max5.size =   2; max5.stride =   2; max5.pad =      0;
    conv6.n =   128; conv6.size =  3; conv6.stride =  1; conv6.pad =     1;
    max7.n =      2; max7.size =   2; max7.stride =   2; max7.pad =      0;
    conv8.n =   256; conv8.size =  3; conv8.stride =  1; conv8.pad =     1;
    max9.n =      2; max9.size =   2; max9.stride =   2; max9.pad =      0;
    conv10.n =  512; conv10.size = 3; conv10.stride = 1; conv10.pad =    1;
    max11.n =     2; max11.size =  2; max11.stride =  1; max11.pad =     0;
    conv12.n = 1024; conv12.size = 3; conv12.stride = 1; conv12.pad =    1;
    conv13.n = 1024; conv13.size = 3; conv13.stride = 1; conv13.pad =    1;
    conv14.n =  125; conv14.size = 1; conv14.stride = 1; conv14.pad =    0;
    region15.n = 5;
    region15.size = 20;

    layer start;
    start.type = START;
    start.output_h = start.output_w = 416;
    start.output_c = 3;
    start.prec = SINGLE;

    setup_layers(&start, &conv0);
    for (int i = 0; i < NLAYERS - 1; i++)
      setup_layers(layers[i], layers[i+1]);
  }


  // Load weights
  {
    size_t max_io = max_size(layers, NLAYERS);
    size_t max_ws = max_workspace(layers, NLAYERS);

    input = safe_malloc(sizeof(float) * max_io);
    output = safe_malloc(sizeof(float) * max_io);
    workspace = safe_malloc(sizeof(float) * max_ws);
    printf("%lu %lu\n", max_io, max_ws);

  }
  
  {
    FILE* fp = fopen(weights, "rb");
    if (!fp) {
      perror(weights);
      return -2;
    }
    load_layers(layers, NLAYERS, fp);
    fclose(fp);
  }

  // Load input image
  for (int image_id = 0 ; images[image_id] ; image_id++)
  {
    char* fn = images[image_id];
    int w, h, c;
    unsigned char* data = stbi_load(fn, &w, &h, &c, 3);
    if (w != conv0.w || h != conv0.h || c != conv0.c)
      {
        printf("Bad image size\n");
        return 0;
      }
    if (!data)
      {
        printf("Image load failed\n");
        return 0;
      }
    int i, j, k;
    float* buf = (float*) input;
    for(k = 0; k < c; ++k)
      for(j = 0; j < h; ++j)
        for(i = 0; i < w; ++i)
          {
            int dst_index = i + w*j + w*h*k;
            int src_index = k + c*i + c*w*j;
            buf[dst_index] = (float)data[src_index]/255.;
          }
    free(data);
    //printf("%u %u %u\n", buf[80000], buf[80001], buf[80002]);
    //cvt_32_16(buf, input, conv0.h*conv0.w*conv0.c);
    //printf("%hu %hu %hu\n", input[80000], input[80001], input[80002]);
    printf("%.3f\n", input[0]);
  
    size_t cycles = rdcycle();

    layer_forward(&conv0, input, output, workspace); swap(&input, &output);
    layer_forward(&bn0, input, output, workspace);
    layer_forward(&bias0, input, output, workspace);
    layer_forward(&leaky0, input, output, workspace);
    layer_forward(&max1, input, output, workspace);

    layer_forward(&conv2, input, output, workspace); swap(&input, &output);
    layer_forward(&bn2, input, output, workspace);
    layer_forward(&bias2, input, output, workspace);
    layer_forward(&leaky2, input, output, workspace);
    layer_forward(&max3, input, output, workspace);

    layer_forward(&conv4, input, output, workspace); swap(&input, &output);
    layer_forward(&bn4, input, output, workspace);
    layer_forward(&bias4, input, output, workspace);
    layer_forward(&leaky4, input, output, workspace);
    layer_forward(&max5, input, output, workspace);

    layer_forward(&conv6, input, output, workspace); swap(&input, &output);
    layer_forward(&bn6, input, output, workspace);
    layer_forward(&bias6, input, output, workspace);
    layer_forward(&leaky6, input, output, workspace);
    layer_forward(&max7, input, output, workspace);

    layer_forward(&conv8, input, output, workspace); swap(&input, &output);
    layer_forward(&bn8, input, output, workspace);
    layer_forward(&bias8, input, output, workspace);
    layer_forward(&leaky8, input, output, workspace);
    layer_forward(&max9, input, output, workspace);

    layer_forward(&conv10, input, output, workspace); swap(&input, &output);
    layer_forward(&bn10, input, output, workspace);
    layer_forward(&bias10, input, output, workspace);
    layer_forward(&leaky10, input, output, workspace);
    layer_forward(&max11, input, output, workspace);

    layer_forward(&conv12, input, output, workspace); swap(&input, &output);
    layer_forward(&bn12, input, output, workspace);
    layer_forward(&bias12, input, output, workspace);
    layer_forward(&leaky12, input, output, workspace);

    layer_forward(&conv13, input, output, workspace); swap(&input, &output);
    layer_forward(&bn13, input, output, workspace);
    layer_forward(&bias13, input, output, workspace);
    layer_forward(&leaky13, input, output, workspace);

    layer_forward(&conv14, input, output, workspace); swap(&input, &output);
    layer_forward(&bias14, input, output, workspace);

    layer_forward(&region15, input, output, workspace);

    cycles = rdcycle() - cycles;
    /* for (int i = 0; i < region15.output_h*region15.output_w*region15.output_c; i++) */
    /*   printf("%d %.3f\n", i, output[i]); */
    printf("cycle: %lu\n", cycles);
  }
  
  return 0;
}
