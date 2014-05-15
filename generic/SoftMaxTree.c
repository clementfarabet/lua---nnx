#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SoftMaxTree.c"
#else

static int nn_(SoftMaxTree_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);  
  THTensor *target = luaT_checkudata(L, 3, torch_Tensor);  
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  long rootId = (long)(luaT_getfieldcheckint(L, 1, "rootId"));
  
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  THTensor *childParent = luaT_getfieldcheckudata(L, 1, "childParent", torch_Tensor);
  
  THTensor *node, *nodeWeight, *nodeBias, *nodeOutput;
  

  long k, i;
  
  luaL_argcheck(L, input->nDimension == 2, 2, "2D(batch mode) tensor expected");
  luaL_argcheck(L, input->size[1] == inputSize, 2, "invalid input size");

  node = THTensor_(new)();
  nodeWeight = THTensor_(new)();
  nodeBias = THTensor_(new)();
  nodeOutput = THTensor_(new)();
  
  THTensor_(resize1d)(output, input->size[0]);
  
  for(i = 0; i < input->size[0], i++)
  {
    long childId = (long)(THTensor_(get1d)(target, i));
    real activation = 

    while (1)
    {
      long parentId = (long)(THTensor_(get1d)(childParent, childId));
      long parentIdx, nChildren;
      
      THTensor_(select)(node, parentChildren, 0, parentId);
      parentIdx = (long)(THTensor_(get1d)(node, 0));
      nChildren = (long)(THTensor_(get1d)(node, 1));
      
      THTensor_(narrow)(nodeWeight, weight, 0, parentIdx, nChildren);
      THTensor_(narrow)(nodeBias, bias, 0, parentIdx, nChildren);
      TH_API void THTensor_(addmv)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *mat,  THTensor *vec);
      
      THTensor_(addmv)(nodeOutput, 1, nodeBias, 1, nodeWeight, input);
      
      
      
      if parentId == rootId) 
      {
        break;
      }
      childId = parentId;
    }
    end
    -- sample channel (one channel per sample)
    local channel = nn.Sequential()
    channel:add(concat)
    channel:add(nn.CMulTable())
    parallel:add(channel)
  
  }
  THTensor_(free)(node);
  THTensor_(free)(nodeWeight);
  THTensor_(free)(nodeBias);
  THTensor_(free)(nodeOutput);
  return 1;
}

static int nn_(SoftMaxTree_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);  
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  long nInputFrame;
  long nOutputFrame;

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  THTensor *gradOutputWindow;
  THTensor *gradInputWindow;
  long k, i;
  
  int dimS = 0; // sequence dimension
  int dimF = 1; // feature dimension
  
  if (gradOutput->nDimension == 3) 
  {
    dimS = 1;
    dimF = 2;
  }
  
  nInputFrame = input->size[dimS];
  nOutputFrame = gradOutput->size[dimS];

  

  return 1;
}

static int nn_(SoftMaxTree_accGradParameters)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);  
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);  
  real scale = luaL_optnumber(L, 4, 1);
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  long nInputFrame;
  long nOutputFrame;

  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor);
  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);

  THTensor *gradOutputWindow;
  THTensor *inputWindow;
  long k, i;
  
  int dimS = 0; // sequence dimension
  int dimF = 1; // feature dimension
  
  if (gradOutput->nDimension == 3) 
  {
    dimS = 1;
    dimF = 2;
  }
  
  nInputFrame = input->size[dimS];
  nOutputFrame = gradOutput->size[dimS];

  input = THTensor_(newContiguous)(input);
  gradOutputWindow = THTensor_(new)();
  inputWindow = THTensor_(new)();
  
  
  THTensor_(free)(input);

  return 0;
}

static const struct luaL_Reg nn_(SoftMaxTree__) [] = {
  {"TemporalConvolution_updateOutput", nn_(SoftMaxTree_updateOutput)},
  {"TemporalConvolution_updateGradInput", nn_(SoftMaxTree_updateGradInput)},
  {"TemporalConvolution_accGradParameters", nn_(SoftMaxTree_accGradParameters)},
  {NULL, NULL}
};

static void nn_(SoftMaxTree_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SoftMaxTree__), "nn");
  lua_pop(L,1);
}

#endif
