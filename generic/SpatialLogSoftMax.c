#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialLogSoftMax.c"
#else

static int nn_(SpatialLogSoftMax_forward)(lua_State *L)
{
  // get all params
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));

  // dims
  int width = input->size[2];
  int height = input->size[1];

  // select planes
  THTensor *input_row = THTensor_(new)();
  THTensor *input_point = THTensor_(new)();
  THTensor *output_row = THTensor_(new)();
  THTensor *output_point = THTensor_(new)();

  // process the whole plane
  int x,y;
  for (y=0; y<height; y++) {
    THTensor_(select)(input_row, input, 1, y);
    THTensor_(select)(output_row, output, 1, y);
    for (x=0; x<width; x++) {
      THTensor_(select)(input_point, input_row, 1, x);
      THTensor_(select)(output_point, output_row, 1, x);
      
      real sum = THLogZero;
      
      TH_TENSOR_APPLY2(real, output_point, real, input_point,           \
                       real z = *input_point_data;                      \
                       *output_point_data = z;                          \
                       sum = THLogAdd(sum, z);)

      THTensor_(add)(output_point, -sum);
    }
  }

  // cleanup
  THTensor_(free)(input_row);
  THTensor_(free)(input_point);
  THTensor_(free)(output_row);
  THTensor_(free)(output_point);
  return 1;
}

static int nn_(SpatialLogSoftMax_backward)(lua_State *L)
{
  // get all params
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));

  // dims
  int width = input->size[2];
  int height = input->size[1];

  // zero gradInput
  THTensor_(zero)(gradInput);

  // select planes
  THTensor *gradOutput_row = THTensor_(new)();
  THTensor *gradOutput_point = THTensor_(new)();
  THTensor *gradInput_row = THTensor_(new)();
  THTensor *gradInput_point = THTensor_(new)();
  THTensor *output_row = THTensor_(new)();
  THTensor *output_point = THTensor_(new)();

  // compute gradients for each point
  int x,y;
  for (y=0; y<height; y++) {
    THTensor_(select)(gradInput_row, gradInput, 1, y);
    THTensor_(select)(gradOutput_row, gradOutput, 1, y);
    THTensor_(select)(output_row, output, 1, y);
    for (x=0; x<width; x++) {
      THTensor_(select)(gradInput_point, gradInput_row, 1, x);
      THTensor_(select)(gradOutput_point, gradOutput_row, 1, x);
      THTensor_(select)(output_point, output_row, 1, x);
      
      real sum = THTensor_(sum)(gradOutput_point);
      
      TH_TENSOR_APPLY3(real, gradInput_point,        \
                       real, gradOutput_point,       \
                       real, output_point,                            \
                       *gradInput_point_data = *gradOutput_point_data - exp(*output_point_data)*sum;);
    }
  }
  
  // cleanup
  THTensor_(free)(gradInput_row);
  THTensor_(free)(gradInput_point);
  THTensor_(free)(gradOutput_row);
  THTensor_(free)(gradOutput_point);
  THTensor_(free)(output_row);
  THTensor_(free)(output_point);
  
  return 1;
}

static const struct luaL_Reg nn_(SpatialLogSoftMax__) [] = {
  {"SpatialLogSoftMax_forward", nn_(SpatialLogSoftMax_forward)},
  {"SpatialLogSoftMax_backward", nn_(SpatialLogSoftMax_backward)},
  {NULL, NULL}
};

static void nn_(SpatialLogSoftMax_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(SpatialLogSoftMax__), "nn");
  lua_pop(L,1);
}

#endif
