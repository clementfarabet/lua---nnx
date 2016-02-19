#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SoftMaxTree.c"
#else

static int nn_(SoftMaxTree_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);  
  THIntTensor *target = (THIntTensor*)luaT_checkudata(L, 3, "torch.IntTensor");  
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  long rootId = (long)(luaT_getfieldcheckint(L, 1, "rootId") - 1);
  long maxFamilyPath = (long)luaT_getfieldcheckint(L, 1, "maxFamilyPath");
  
  THIntTensor *childParent = (THIntTensor*)luaT_getfieldcheckudata(L, 1, "childParent", "torch.IntTensor");
  THIntTensor *parentChildren = (THIntTensor*)luaT_getfieldcheckudata(L, 1, "parentChildren", "torch.IntTensor");
  
  THTensor *linearOutput = luaT_getfieldcheckudata(L, 1, "_nodeBuffer", torch_Tensor);
  THTensor *logsoftOutput = luaT_getfieldcheckudata(L, 1, "_multiBuffer", torch_Tensor);
  
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  
  THIntTensor *node;
  THTensor *nodeWeight, *nodeBias, *nodeOutput, *nodeInput, *nodeInter;
  real *input_data, *output_data;

  long i, d;
  
  luaL_argcheck(L, input->nDimension == 2, 2, "2D(batch mode) tensor expected");
  luaL_argcheck(L, input->size[1] == inputSize, 2, "invalid input size");

  node = THIntTensor_new();
  nodeWeight = THTensor_(new)();
  nodeBias = THTensor_(new)();
  nodeOutput = THTensor_(new)();
  nodeInput = THTensor_(new)();
  nodeInter = THTensor_(new)();
  
  THTensor_(resize1d)(output, input->size[0]);
  
  for(i = 0; i < input->size[0]; i++)
  {
    long n = 0;
    long childId = (long)(THIntTensor_get1d(target, i)) - 1;
    accreal narrowsum = 0;
    THTensor_(select)(nodeInput, input, 0, i);
    while(1)
    {
      long parentId, parentIdx, childIdx, nChildren;
      /* get next Node in Tree */
      THIntTensor_select(node, childParent, 0, childId);
      parentId = (long)(THIntTensor_get1d(node, 0)) - 1;
      childIdx = (long)(THIntTensor_get1d(node, 1)) - 1;
      
      luaL_argcheck(L, parentId != -2, 2, "Non-root node has no parent in tree.");
      
      THIntTensor_select(node, parentChildren, 0, parentId);
      parentIdx = (long)(THIntTensor_get1d(node, 0)) - 1;
      nChildren = (long)(THIntTensor_get1d(node, 1));
  
      /* Linear */
      THTensor_(narrow)(nodeWeight, weight, 0, parentIdx, nChildren);
      THTensor_(narrow)(nodeBias, bias, 0, parentIdx, nChildren);
      THTensor_(narrow)(nodeOutput, linearOutput, 0, 0, nChildren);
      
      THTensor_(addmv)(nodeOutput, 1, nodeBias, 1, nodeWeight, nodeInput);
      
      /* LogSoftMax */
      THTensor_(set)(nodeInter, nodeOutput);
      THTensor_(narrow)(nodeOutput, logsoftOutput, 0, maxFamilyPath*i + n, nChildren);
      
      input_data = THTensor_(data)(nodeInter);
      output_data = THTensor_(data)(nodeOutput);
      
      accreal logsum = 0;
      real maxInput = -THInf;
      
      for(d = 0; d < nChildren; d++)
        maxInput = THMax(maxInput, input_data[d]);

      for(d = 0; d < nChildren; d++)
        logsum += THExpMinusApprox(maxInput-input_data[d]);
      logsum = maxInput + log(logsum);

      for(d = 0; d < nChildren; d++)
        output_data[d] = input_data[d] - logsum;
        
      /* Narrow */
      THTensor_(set)(nodeInter, nodeOutput);
      THTensor_(narrow)(nodeOutput, nodeInter, 0, childIdx, 1); 
      
      /* CAddTable (without log, would have been CMulTable) */
      narrowsum += THTensor_(get1d)(nodeOutput, 0);
      n += nChildren;
      /* Break when root is reached */
      if (parentId == rootId) 
      {
        break;
      }
      childId = parentId;
    }
    THTensor_(set1d)(output, i, narrowsum);  
  }
  
  THIntTensor_free(node);
  THTensor_(free)(nodeWeight);
  THTensor_(free)(nodeBias);
  THTensor_(free)(nodeOutput);
  THTensor_(free)(nodeInput);
  THTensor_(free)(nodeInter);
  return 1;
}

static int nn_(SoftMaxTree_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);  
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);  
  THIntTensor *target = (THIntTensor*)luaT_checkudata(L, 4, "torch.IntTensor");  
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  long rootId = (long)(luaT_getfieldcheckint(L, 1, "rootId") - 1);
  long maxFamilyPath = (long)luaT_getfieldcheckint(L, 1, "maxFamilyPath");
  
  THIntTensor *childParent = (THIntTensor*)luaT_getfieldcheckudata(L, 1, "childParent", "torch.IntTensor");
  THIntTensor *parentChildren = (THIntTensor*)luaT_getfieldcheckudata(L, 1, "parentChildren", "torch.IntTensor");
  
  THTensor *logsoftOutput = luaT_getfieldcheckudata(L, 1, "_multiBuffer", torch_Tensor);
  
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "_gradInput", torch_Tensor);
  
  THIntTensor *node;
  THTensor *nodeWeight, *nodeOutput;
  THTensor *nodeGradInput, *weightTranspose;
  real *output_data;

  long i, d;
  
  luaL_argcheck(L, input->nDimension == 2, 2, "2D(batch mode) tensor expected");
  luaL_argcheck(L, input->size[1] == inputSize, 2, "invalid input size");
  
  luaL_argcheck(L, gradOutput->nDimension == 1, 2, "1D tensor expected");

  node = THIntTensor_new();
  nodeWeight = THTensor_(new)();
  nodeOutput = THTensor_(new)();
  nodeGradInput = THTensor_(new)();
  weightTranspose = THTensor_(new)();
  
  THTensor_(transpose)(weightTranspose, weight, 0, 1);
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);
  
  for(i = 0; i < input->size[0]; i++)
  {
    long n = 0;
    long childId = (long)(THIntTensor_get1d(target, i)) - 1;
    real grad = THTensor_(get1d)(gradOutput, i);
    
    THTensor_(select)(nodeGradInput, gradInput, 0, i);
    
    while(1)
    {
      long parentId, parentIdx, childIdx, nChildren;
      /* get next Node in Tree */
      THIntTensor_select(node, childParent, 0, childId);
      parentId = (long)(THIntTensor_get1d(node, 0)) - 1;
      childIdx = (long)(THIntTensor_get1d(node, 1)) - 1;
      
      luaL_argcheck(L, parentId != -2, 2, "Non-root node has no parent in tree.");
      
      THIntTensor_select(node, parentChildren, 0, parentId);
      parentIdx = (long)(THIntTensor_get1d(node, 0)) - 1;
      nChildren = (long)(THIntTensor_get1d(node, 1));
      
      luaL_argcheck(L, logsoftOutput->size[0] >= n+nChildren, 2, \
        "Backward performed on different inputs than last forward");
      
      /* CAddTable + Narrow + LogSoftMax */
      THTensor_(narrow)(nodeOutput, logsoftOutput, 0, maxFamilyPath*i + n, nChildren);
      
      output_data = THTensor_(data)(nodeOutput);

      for(d = 0; d < nChildren; d++)
        output_data[d] = -exp(output_data[d])*grad;
      output_data[childIdx] += grad;

  
      /* Linear */
      THTensor_(narrow)(nodeWeight, weightTranspose, 1, parentIdx, nChildren);
      
      THTensor_(addmv)(nodeGradInput, 1, nodeGradInput, 1, nodeWeight, nodeOutput);
      
      n += nChildren;
      /* Break when root is reached */
      if (parentId == rootId) 
      {
        break;
      }
      childId = parentId;
    }
  }
  
  THIntTensor_free(node);
  THTensor_(free)(nodeWeight);
  THTensor_(free)(nodeOutput);
  THTensor_(free)(nodeGradInput);
  THTensor_(free)(weightTranspose);
  return 1;
}

static int nn_(SoftMaxTree_accGradParameters)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);  
  THIntTensor *target = (THIntTensor*)luaT_checkudata(L, 4, "torch.IntTensor");  
  real scale = luaL_optnumber(L, 5, 1);
  long rootId = (long)(luaT_getfieldcheckint(L, 1, "rootId") - 1);
  long maxFamilyPath = (long)luaT_getfieldcheckint(L, 1, "maxFamilyPath");
  
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  THIntTensor *childParent = (THIntTensor*)luaT_getfieldcheckudata(L, 1, "childParent", "torch.IntTensor");
  THIntTensor *parentChildren = (THIntTensor*)luaT_getfieldcheckudata(L, 1, "parentChildren", "torch.IntTensor");
  
  THTensor *linearGradOutput = luaT_getfieldcheckudata(L, 1, "_multiBuffer", torch_Tensor);;
  
  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor);
  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);
  
  lua_getfield(L, 1, "updates");
  
  THIntTensor *node;
  THTensor *nodeGradWeight, *nodeGradBias, *nodeInput, *nodeGradOutput;

  long i;
  
  luaL_argcheck(L, input->nDimension == 2, 2, "2D(batch mode) tensor expected");
  luaL_argcheck(L, input->size[1] == inputSize, 2, "invalid input size");
  
  node = THIntTensor_new();
  nodeGradWeight = THTensor_(new)();
  nodeGradBias = THTensor_(new)();
  nodeGradOutput = THTensor_(new)();
  nodeInput = THTensor_(new)();
  
  for(i = 0; i < input->size[0]; i++)
  {
    long n = 0;
    long childId = (long)(THIntTensor_get1d(target, i)) - 1;
    THTensor_(select)(nodeInput, input, 0, i);
    
    while(1)
    {
      long parentId, parentIdx, childIdx, nChildren;
      double count;
      /* get next Node in Tree */
      THIntTensor_select(node, childParent, 0, childId);
      parentId = (long)(THIntTensor_get1d(node, 0)) - 1;
      childIdx = (long)(THIntTensor_get1d(node, 1)) - 1;
      
      luaL_argcheck(L, parentId != -2, 2, "Non-root node has no parent in tree.");
      
      THIntTensor_select(node, parentChildren, 0, parentId);
      parentIdx = (long)(THIntTensor_get1d(node, 0)) - 1;
      nChildren = (long)(THIntTensor_get1d(node, 1));
      
      THTensor_(narrow)(nodeGradOutput, linearGradOutput, 0, maxFamilyPath*i + n, nChildren);
      THTensor_(narrow)(nodeGradWeight, gradWeight, 0, parentIdx, nChildren);
      THTensor_(narrow)(nodeGradBias, gradBias, 0, parentIdx, nChildren);
      
      THTensor_(addr)(nodeGradWeight, 1, nodeGradWeight, scale, nodeGradOutput, nodeInput);
      THTensor_(cadd)(nodeGradBias, nodeGradBias, scale, nodeGradOutput);
      
      /* updates will contain parentId (key) sum of scales (value)*/
      lua_pushinteger(L, (int)(parentId+1));
      lua_gettable(L, -2);
      count = lua_tonumber(L, -1) + scale;
      lua_pop(L, 1);
      
      lua_pushinteger(L, (int)(parentId+1)); /* key */
      lua_pushnumber(L, count); /* value */
      lua_settable(L, -3);
      
      n += nChildren;
      /* Break when root is reached */
      if (parentId == rootId) 
      {
        break;
      }
      childId = parentId;
    }
  }
  
  THIntTensor_free(node);
  THTensor_(free)(nodeGradWeight);
  THTensor_(free)(nodeGradBias);
  THTensor_(free)(nodeGradOutput);
  THTensor_(free)(nodeInput);
    
  return 0;
}

static const struct luaL_Reg nn_(SoftMaxTree__) [] = {
  {"SoftMaxTree_updateOutput", nn_(SoftMaxTree_updateOutput)},
  {"SoftMaxTree_updateGradInput", nn_(SoftMaxTree_updateGradInput)},
  {"SoftMaxTree_accGradParameters", nn_(SoftMaxTree_accGradParameters)},
  {NULL, NULL}
};

static void nn_(SoftMaxTree_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SoftMaxTree__), "nn");
  lua_pop(L,1);
}

#endif
