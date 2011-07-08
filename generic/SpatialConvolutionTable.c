#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialConvolutionTable.c"
#else

static int nn_(SpatialConvolutionTable_forward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  THTensor *connTable = luaT_getfieldcheckudata(L, 1, "connTable", torch_(Tensor_id));
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));

  luaL_argcheck(L, input->nDimension == 3, 2, "3D tensor expected");
  luaL_argcheck(L, input->size[0] == nInputPlane, 2, "invalid number of input planes");
  luaL_argcheck(L, input->size[2] >= kW && input->size[1] >= kH, 2, "input image smaller than kernel size");

  THTensor_(resize3d)(output, nOutputPlane,
                      (input->size[1] - kH) / dH + 1, 
                      (input->size[2] - kW) / dW + 1);

  THTensor *inputPlane = THTensor_(new)();
  THTensor *weightPlane = THTensor_(new)();
  THTensor *outputPlane = THTensor_(new)();

  /* Add bias */
  int k;
  for (k = 0; k < nOutputPlane; k++)
  {
    THTensor_(select)(outputPlane,output,0,k);
    THTensor_(fill)(outputPlane, THTensor_(get1d)(bias, k));
  }

  /* Convolve all maps */
  int nkernel = connTable->size[0];
  for (k = 0; k < nkernel; k++)
  {
    int outplaneid = (int)THTensor_(get2d)(connTable,k,1)-1;
    int inplaneid = (int)THTensor_(get2d)(connTable,k,0)-1;

    /* Get input, output and kernel*/
    THTensor_(select)(outputPlane, output, 0, outplaneid);
    THTensor_(select)(inputPlane, input, 0, inplaneid);
    THTensor_(select)(weightPlane, weight, 0, k);

    /* Convolve */
    THLab_(conv2Dmul)(outputPlane, 1.0, inputPlane, weightPlane, dH, dW, "vx");
  }

  /* Cleanup */
  THTensor_(free)(inputPlane);
  THTensor_(free)(weightPlane);
  THTensor_(free)(outputPlane);

  return 1;
}

static int nn_(SpatialConvolutionTable_backward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  THTensor *connTable = luaT_getfieldcheckudata(L, 1, "connTable", torch_(Tensor_id));
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_(Tensor_id));
  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));

  THTensor *gradInputPlane = THTensor_(new)();
  THTensor *inputPlane = THTensor_(new)();
  THTensor *gradOutputPlane = THTensor_(new)();
  THTensor *weightPlane = THTensor_(new)();
  THTensor *gradWeightPlane = THTensor_(new)();
  
  /* Resize/Zero */
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  /* gradients wrt bias */
  int i, k;
  real *gradBias_data = THTensor_(data)(gradBias);
  for(k = 0; k < nOutputPlane; k++)
  {
    THTensor_(select)(gradOutputPlane, gradOutput, 0, k);
    gradBias_data[k] += THTensor_(sum)(gradOutputPlane);
  }

  int nkernel = connTable->size[0];
  for(k = 0; k < nkernel; k++)
  {
    int outplaneid = (int)THTensor_(get2d)(connTable,k,1)-1;
    int inplaneid = (int)THTensor_(get2d)(connTable,k,0)-1;
    
    /* Select all planes */
    THTensor_(select)(inputPlane, input, 0, inplaneid);
    THTensor_(select)(gradInputPlane, gradInput, 0, inplaneid);
    THTensor_(select)(gradOutputPlane, gradOutput, 0, outplaneid);
    THTensor_(select)(weightPlane, weight, 0, k);
    THTensor_(select)(gradWeightPlane, gradWeight, 0, k);

    /* Gradient to kernel */
    THTensor_(resize3d)(inputPlane, 1, inputPlane->size[0], inputPlane->size[1]);
    THTensor_(resize3d)(gradOutputPlane, 1, gradOutputPlane->size[0], gradOutputPlane->size[1]);
    THTensor_(resize4d)(gradWeightPlane, 1, 1, gradWeightPlane->size[0], gradWeightPlane->size[1]);
    THLab_(conv2DRevger)(gradWeightPlane, 1.0, inputPlane, gradOutputPlane, dH, dW);
    
    /* Gradient to input */
    THTensor_(resize3d)(gradInputPlane, 1, gradInputPlane->size[0], gradInputPlane->size[1]);
    THTensor_(resize4d)(weightPlane, 1, 1, weightPlane->size[0], weightPlane->size[1]);
    THLab_(conv2Dmv)(gradInputPlane, 1.0, gradOutputPlane, weightPlane, dH, dW, "fx");
  }

  THTensor_(free)(gradInputPlane);
  THTensor_(free)(inputPlane);
  THTensor_(free)(gradOutputPlane);
  THTensor_(free)(weightPlane);
  THTensor_(free)(gradWeightPlane);

  return 1;
}

static const struct luaL_Reg nn_(SpatialConvolutionTable__) [] = {
  {"SpatialConvolutionTable_forward", nn_(SpatialConvolutionTable_forward)},
  {"SpatialConvolutionTable_backward", nn_(SpatialConvolutionTable_backward)},
  {NULL, NULL}
};

static void nn_(SpatialConvolutionTable_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(SpatialConvolutionTable__), "nn");
  lua_pop(L,1);
}

#endif
