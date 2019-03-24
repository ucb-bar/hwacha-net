#include <stdio.h>
#include <string.h>
#include "util.h"

enum arg_type { OPTION, IMAGE, WEIGHTS };

int parse_args(int argc, char** argv, char* weights, char* images[])
{
  enum arg_type type = argc == 2 ? IMAGE : OPTION;
  int image_id = 0;
  for (int i = 1 ; i < argc ; i++) {
    switch(type) {
      case OPTION:
        if (strcmp(argv[i], "-w") == 0)
          type = WEIGHTS;
        else if (strcmp(argv[i], "-i") == 0)
          type = IMAGE;
        break;
      case IMAGE:
        if (image_id < 100) {
          size_t n = strlen(argv[i]);
          images[image_id] = safe_malloc((n+1)*sizeof(char));
          strcpy(images[image_id++], argv[i]);
        }
        type = OPTION;
        break;
      case WEIGHTS:
        strcpy(weights, argv[i]);
        type = OPTION;
        break;
      default:
        break;
    }
  }
  return image_id > 0;
}
