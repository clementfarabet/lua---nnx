#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialMatching.c"
#else

#define square(x) ((x)*(x))
#define max(x,y) (((x)>(y)) ? (x) : (y))
#define min(x,y) (((x)>(y)) ? (y) : (x))

static int nn_(SpatialMatching_updateOutput)(lua_State *L)
{
  // get all params
  THTensor *input1 = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *input2 = luaT_checkudata(L, 3, torch_(Tensor_id));
  int maxw = luaT_getfieldcheckint(L, 1, "maxw");
  int maxh = luaT_getfieldcheckint(L, 1, "maxh");
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));

  // dims
  int iwidth = input1->size[2];
  int iheight = input1->size[1];
  int ichannels = input1->size[0];
  int owidth = iwidth;
  int oheight = iheight;

  // zero output
  THTensor_(fill)(output, 1e30);

  // get strides
  long *i1s = input1->stride;
  long *i2s = input2->stride;
  long *os  = output->stride;

  // get pointers
  real *input1_p = THTensor_(data)(input1);
  real *input2_p = THTensor_(data)(input2);
  real *output_p = THTensor_(data)(output);

  // compute output
  int x1,y1,x2,y2,k;
#pragma omp parallel for private(x1,y1,x2,y2,k)
  for (y1=0; y1<oheight; y1++) {
    for (x1=0; x1<owidth; x1++) {
      for (y2=max(0,(y1-ceil(maxh/2))); y2<min(oheight,y1+floor(maxh/2)+1); y2++) {
        for (x2=max(0,(x1-ceil(maxw/2))); x2<min(owidth,x1+floor(maxw/2)+1); x2++) {
          real dist = 0;
          for (k=0; k<ichannels; k++) {
            dist += square(input1_p[k*i1s[0] + y1*i1s[1] + x1*i1s[2]] - input2_p[k*i2s[0] + y2*i2s[1] + x2*i2s[2]]);
          }
          long dy = y2-y1 + ceil(maxh/2);
          long dx = x2-x1 + ceil(maxw/2);
          output_p[y1*os[0] + x1*os[1] + dy*os[2] + dx*os[3]] = dist;
        }
      }
    }
  }

  // done
  return 1;
}

static int nn_(SpatialMatching_updateGradInput)(lua_State *L)
{
  // get all params
  THTensor *input1 = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *input2 = luaT_checkudata(L, 3, torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));
  int maxw = luaT_getfieldcheckint(L, 1, "maxw");
  int maxh = luaT_getfieldcheckint(L, 1, "maxh");

  // dims
  int iwidth = input1->size[2];
  int iheight = input1->size[1];
  int ichannels = input1->size[0];
  int owidth = gradOutput->size[2];
  int oheight = gradOutput->size[1];
  int ochannels = gradOutput->size[0];

  // resize gradInput
  THTensor_(zero)(gradInput);

  // compute gradients
  int x,y,k;
  for (k=0; k<ichannels; k++) {
    for (y=0; y<oheight; y++) {
      for (x=0; x<owidth; x++) {
      }
    }
  }

  // done
  return 1;
}

static const struct luaL_Reg nn_(SpatialMatching__) [] = {
  {"SpatialMatching_updateOutput", nn_(SpatialMatching_updateOutput)},
  {"SpatialMatching_updateGradInput", nn_(SpatialMatching_updateGradInput)},
  {NULL, NULL}
};

static void nn_(SpatialMatching_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(SpatialMatching__), "nn");
  lua_pop(L,1);
}

#endif
