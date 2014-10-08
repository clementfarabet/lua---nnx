#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialReSampling.c"
#else

#ifndef MAX
#define MAX(a,b) ( ((a)>(b)) ? (a) : (b) )
#endif
#ifndef MIN
#define MIN(a,b) ( ((a)<(b)) ? (a) : (b) )
#endif

static int nn_(SpatialReSampling_updateOutput)(lua_State *L)
{
  // get all params
  THTensor *input_ = luaT_checkudata(L, 2, torch_Tensor);
  int owidth = luaT_getfieldcheckint(L, 1, "owidth");
  int oheight = luaT_getfieldcheckint(L, 1, "oheight");
  THTensor *output_ = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  // check dims
  luaL_argcheck(L, (input_->nDimension == 3) || (input_->nDimension == 4), 2, "3D or 4D tensor expected");

  // dims
  int channelDim = 0;
  int batchSize = 1;
  if (input_->nDimension == 4){
    channelDim = 1;
    batchSize = input_->size[0];
  }
  
  int iwidth = input_->size[channelDim + 2];
  int iheight = input_->size[channelDim + 1];
  int ochannels = input_->size[channelDim];

  // resize output
  if (input_->nDimension == 3)
    THTensor_(resize3d)(output_, ochannels, oheight, owidth);
  else
    THTensor_(resize4d)(output_, batchSize, ochannels, oheight, owidth);
  
  // select example
  THTensor *output = THTensor_(newWithTensor)(output_);
  THTensor *input = THTensor_(newWithTensor)(input_);
  
  // select planes
  THTensor *outputPlane = THTensor_(new)();
  THTensor *inputPlane = THTensor_(new)();

  // mapping ratios
  float wratio = (float)(iwidth-1) / (owidth-1);
  float hratio = (float)(iheight-1) / (oheight-1);

  int b;
  for (b=0; b<batchSize; b++) {
    if (input_->nDimension == 4) 
    {
      THTensor_(select)(input, input_, 0, b);
      THTensor_(select)(output, output_, 0, b);
    }  
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
          // subpixel position:
          float ix = wratio*x;
          float iy = hratio*y;

          // 4 nearest neighbors:
          float ix_nw = floor(ix);
          float iy_nw = floor(iy);
          float ix_ne = ix_nw + 1;
          float iy_ne = iy_nw;
          float ix_sw = ix_nw;
          float iy_sw = iy_nw + 1;
          float ix_se = ix_nw + 1;
          float iy_se = iy_nw + 1;

          // get surfaces to each neighbor:
          float se = (ix-ix_nw)*(iy-iy_nw);
          float sw = (ix_ne-ix)*(iy-iy_ne);
          float ne = (ix-ix_sw)*(iy_sw-iy);
          float nw = (ix_se-ix)*(iy_se-iy);

          // weighted sum of neighbors:
          double sum = THTensor_(get2d)(inputPlane, iy_nw, ix_nw) * nw
            + THTensor_(get2d)(inputPlane, iy_ne, MIN(ix_ne,iwidth-1)) * ne
            + THTensor_(get2d)(inputPlane, MIN(iy_sw,iheight-1), ix_sw) * sw
            + THTensor_(get2d)(inputPlane, MIN(iy_se,iheight-1), MIN(ix_se,iwidth-1)) * se;

          // set output
          THTensor_(set2d)(outputPlane, y, x, sum);
        }
      }
    }
  }

  // cleanup
  THTensor_(free)(inputPlane);
  THTensor_(free)(outputPlane);
  THTensor_(free)(input);
  THTensor_(free)(output);
  return 1;
}

static int nn_(SpatialReSampling_updateGradInput)(lua_State *L)
{
  // get all params
  THTensor *input_ = luaT_checkudata(L, 2, torch_Tensor);  
  THTensor *gradOutput_ = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *gradInput_ = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  // dim
  int channelDim = 0;
  int batchSize = 1;
  if (input_->nDimension == 4){
    channelDim = 1;
    batchSize = input_->size[0];
  }
  
  int iwidth = input_->size[channelDim+2];
  int iheight = input_->size[channelDim+1];
  int ichannels = input_->size[channelDim];
  int owidth = gradOutput_->size[channelDim+2];
  int oheight = gradOutput_->size[channelDim+1];
  int ochannels = gradOutput_->size[channelDim];

  // resize gradInput
  if (input_->nDimension == 3)
    THTensor_(resize3d)(gradInput_, ichannels, iheight, iwidth);
  else
    THTensor_(resize4d)(gradInput_, batchSize, ichannels, iheight, iwidth);
  THTensor_(zero)(gradInput_);
  
  // select example
  THTensor *gradOutput = THTensor_(newWithTensor)(gradOutput_);
  THTensor *gradInput = THTensor_(newWithTensor)(gradInput_);

  // select planes
  THTensor *gradOutputPlane = THTensor_(new)();
  THTensor *gradInputPlane = THTensor_(new)();

  // mapping ratios
  float wratio = (float)(iwidth-1) / (owidth-1);
  float hratio = (float)(iheight-1) / (oheight-1);

  int b;
  for (b=0; b<batchSize; b++) {
    if (input_->nDimension == 4) 
    {
      THTensor_(select)(gradInput, gradInput_, 0, b);
      THTensor_(select)(gradOutput, gradOutput_, 0, b);
    }  
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
          // subpixel position:
          float ix = wratio*x;
          float iy = hratio*y;

          // 4 nearest neighbors:
          float ix_nw = floor(ix);
          float iy_nw = floor(iy);
          float ix_ne = ix_nw + 1;
          float iy_ne = iy_nw;
          float ix_sw = ix_nw;
          float iy_sw = iy_nw + 1;
          float ix_se = ix_nw + 1;
          float iy_se = iy_nw + 1;

          // get surfaces to each neighbor:
          float se = (ix-ix_nw)*(iy-iy_nw);
          float sw = (ix_ne-ix)*(iy-iy_ne);
          float ne = (ix-ix_sw)*(iy_sw-iy);
          float nw = (ix_se-ix)*(iy_se-iy);

          // output gradient
          double ograd = THTensor_(get2d)(gradOutputPlane, y, x);

          // accumulate gradient
          THTensor_(set2d)(gradInputPlane, iy_nw, ix_nw,
                           THTensor_(get2d)(gradInputPlane, iy_nw, ix_nw) + nw * ograd);
          THTensor_(set2d)(gradInputPlane, iy_ne, MIN(ix_ne,iwidth-1),
                           THTensor_(get2d)(gradInputPlane, iy_ne, MIN(ix_ne,iwidth-1)) + ne * ograd);
          THTensor_(set2d)(gradInputPlane, MIN(iy_sw,iheight-1), ix_sw, 
                           THTensor_(get2d)(gradInputPlane, MIN(iy_sw,iheight-1), ix_sw) + sw * ograd);
          THTensor_(set2d)(gradInputPlane, MIN(iy_se,iheight-1), MIN(ix_se,iwidth-1),
                           THTensor_(get2d)(gradInputPlane, MIN(iy_se,iheight-1), MIN(ix_se,iwidth-1)) + se * ograd);
        }
      }
    }
  }

  // cleanup
  THTensor_(free)(gradInputPlane);
  THTensor_(free)(gradOutputPlane);
  THTensor_(free)(gradInput);
  THTensor_(free)(gradOutput);
  return 1;
}

static const struct luaL_Reg nn_(SpatialReSampling__) [] = {
  {"SpatialReSampling_updateOutput", nn_(SpatialReSampling_updateOutput)},
  {"SpatialReSampling_updateGradInput", nn_(SpatialReSampling_updateGradInput)},
  {NULL, NULL}
};

static void nn_(SpatialReSampling_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SpatialReSampling__), "nn");
  lua_pop(L,1);
}

#endif
