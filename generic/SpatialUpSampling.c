#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialUpSampling.c"
#else

static int nn_(SpatialUpSampling_forward)(lua_State *L)
{
  // get all params
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));

  // dims
  int iwidth = input->size[2];
  int iheight = input->size[1];
  int ochannels = input->size[0];
  int owidth = iwidth * dW;
  int oheight = iheight * dH;

  // select planes
  THTensor *outputPlane = THTensor_(new)();
  THTensor *inputPlane = THTensor_(new)();

  // resample each plane
  int k;
  for (k=0; k<ochannels; k++) {
    // get planes
    THTensor_(select)(inputPlane, input, 0, k);
    THTensor_(select)(outputPlane, output, 0, k);

    // for each plane, resample
    int x,y;
    for (y=0; y<oheight; y++) {
      for (x=0; x<owidth; x++) {
        // input positions (floored)
        int ix = x/dW;
        int iy = y/dH;

        // set output
        THTensor_(set2d)(outputPlane, y, x, THTensor_(get2d)(inputPlane, iy, ix));
      }
    }
  }
  // cleanup
  THTensor_(free)(inputPlane);
  THTensor_(free)(outputPlane);
  return 1;
}

static int nn_(SpatialUpSampling_backward)(lua_State *L)
{
  // get all params
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  // dims
  int iwidth = input->size[2];
  int iheight = input->size[1];
  int ichannels = input->size[0];
  int owidth = gradOutput->size[2];
  int oheight = gradOutput->size[1];
  int ochannels = gradOutput->size[0];

  // resize gradInput
  THTensor_(zero)(gradInput);

  // select planes
  THTensor *gradOutputPlane = THTensor_(new)();
  THTensor *gradInputPlane = THTensor_(new)();

  // compute gradients for each plane
  int k;
  for (k=0; k<ochannels; k++) {
    // get planes
    THTensor_(select)(gradInputPlane, gradInput, 0, k);
    THTensor_(select)(gradOutputPlane, gradOutput, 0, k);

    // for each plane, resample
    int x,y;
    for (y=0; y<oheight; y++) {
      for (x=0; x<owidth; x++) {
        // input positions (floored)
        int ix = x/dW;
        int iy = y/dH;

        // output gradient
        double ograd = THTensor_(get2d)(gradOutputPlane, y, x);

        // accumulate gradient
        THTensor_(set2d)(gradInputPlane, iy, ix, THTensor_(get2d)(gradInputPlane, iy, ix) + ograd);
      }
    }
  }

  // cleanup
  THTensor_(free)(gradInputPlane);
  THTensor_(free)(gradOutputPlane);
  return 1;
}

static const struct luaL_Reg nn_(SpatialUpSampling__) [] = {
  {"SpatialUpSampling_forward", nn_(SpatialUpSampling_forward)},
  {"SpatialUpSampling_backward", nn_(SpatialUpSampling_backward)},
  {NULL, NULL}
};

static void nn_(SpatialUpSampling_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(SpatialUpSampling__), "nn");
  lua_pop(L,1);
}

#endif
