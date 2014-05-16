#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SoftMaxTree.c"
#else

static int nn_(SoftMaxTree_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);  
  THIntTensor *target = (THIntTensor*)(luaT_checkudata(L, 3, torch_Tensor));  
  int inputSize = luaT_getfieldcheckint(L, 1, "inputSize");
  long rootId = (long)(luaT_getfieldcheckint(L, 1, "rootId"));
  
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  THIntTensor *childParent = (THIntTensor*)(luaT_getfieldcheckudata(L, 1, "childParent", torch_Tensor));
  THIntTensor *parentChildren = (THIntTensor*)(luaT_getfieldcheckudata(L, 1, "parentChildren", torch_Tensor));
  
  //THTensor *nodes = luaT_getfieldcheckudata(L, 1, "_nodes", torch_Tensor);
  THTensor *linearOutput = luaT_getfieldcheckudata(L, 1, "_linearOutput", torch_Tensor);
  THTensor *logsoftOutput = luaT_getfieldcheckudata(L, 1, "_logSoftMaxOutput", torch_Tensor);
  THTensor *narrowOutput = luaT_getfieldcheckudata(L, 1, "_narrowOutput", torch_Tensor);
  
  THIntTensor *node;
  THTensor *nodeWeight, *nodeBias, *nodeOutput, *nodeInput;
  real *input_data, *output_data;

  long i, d;
  long n = 0;
  
  luaL_argcheck(L, input->nDimension == 2, 2, "2D(batch mode) tensor expected");
  luaL_argcheck(L, input->size[1] == inputSize, 2, "invalid input size");

  node = THIntTensor_new();
  nodeWeight = THTensor_(new)();
  nodeBias = THTensor_(new)();
  nodeOutput = THTensor_(new)();
  nodeInput = THTensor_(new)();
  
  THTensor_(resize1d)(output, input->size[0]);
  
  for(i = 0; i < input->size[0]; i++)
  {
    long childId = (long)(THIntTensor_get1d(target, i));
    accreal narrowsum = 0;
    THTensor_(select)(nodeInput, input, 0, i);

    while(1)
    {
      long parentId, parentIdx, childIdx, nChildren;
      /* get next Node in Tree */
      THIntTensor_select(node, childParent, 0, childId);
      parentId = (long)(THIntTensor_get1d(node, 0));
      childIdx = (long)(THIntTensor_get1d(node, 1));
      
      luaL_argcheck(L, parentId != -1, 2, "Non-root node has no parent in tree.");
      
      THIntTensor_select(node, parentChildren, 0, parentId);
      parentIdx = (long)(THIntTensor_get1d(node, 0));
      nChildren = (long)(THIntTensor_get1d(node, 1));
      
      // we use these to keep intermediate results for later backprop
      if (linearOutput->size[0] < n+nChildren)
      {
        THTensor_(resize1d)(linearOutput, n+nChildren);
        THTensor_(resize1d)(logsoftOutput, n+nChildren);
      }
      
      /* Linear */
      THTensor_(narrow)(nodeWeight, weight, 0, parentIdx, nChildren);
      THTensor_(narrow)(nodeBias, bias, 0, parentIdx, nChildren);
      THTensor_(narrow)(nodeOutput, linearOutput, 0, n, nChildren);
      
      THTensor_(addmv)(nodeOutput, 1, nodeBias, 1, nodeWeight, nodeInput);
      
      /* LogSoftMax */
      THTensor_(set)(nodeInput, nodeOutput);
      THTensor_(narrow)(nodeOutput, logsoftOutput, 0, n, nChildren);
      
      input_data = THTensor_(data)(nodeInput);
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
      THTensor_(set)(nodeInput, nodeOutput);
      THTensor_(narrow)(nodeOutput, nodeInput, 0, childIdx, 1); //we might have to store childIdx in backprop
        
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
  return 1;
}

static int nn_(SoftMaxTree_updateGradInput)(lua_State *L)
{

  

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
