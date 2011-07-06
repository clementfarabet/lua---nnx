local SpatialMaxPooling, parent = torch.class('nn.SpatialMaxPooling', 'nn.Module')

local help_desc =
[[Applies a 2D sub-sampling over an input image composed of
several input planes. The input tensor in forward(input) is 
expected to be a 3D tensor (width x height x nInputPlane). 
The number of output planes will be the same as nInputPlane.

Compared to the nn.SpatialSubSampling module, a max operator is
used to pool values in the kWxkH input neighborhood.

Note that depending of the size of your kernel, several 
(of the last) columns or rows of the input image might be lost. 
It is up to the user to add proper padding in images.

If the input image is a 3D tensor width x height x nInputPlane, 
the output image size will be owidth x oheight x nInputPlane where

owidth  = (width  - kW) / dW + 1
oheight = (height - kH) / dH + 1 .

The parameters of the sub-sampling can be found in self.weight 
(Tensor of size nInputPlane) and self.bias (Tensor of size nInputPlane). 
The corresponding gradients can be found in self.gradWeight and self.gradBias.

The output value of the layer can be precisely described as:

output[i][j][k] = bias[k]
   + weight[k] sum_{s=1}^kW sum_{t=1}^kH input[dW*(i-1)+s][dH*(j-1)+t][k] ]]

function SpatialMaxPooling:__init(kW, kH, dW, dH)
   parent.__init(self)

   -- usage
   if not kW or not kH then
      error(xlua.usage('nn.SpatialMaxPooling', help_desc, nil,
                          {type='number', help='kernel width', req=true},
                          {type='number', help='kernel height', req=true},
                          {type='number', help='stride width [default = kernel width]'},
                          {type='number', help='stride height [default = kernel height]'}))
   end

   dW = dW or kW
   dH = dH or kH
   
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH

   self.indices = torch.Tensor()
end

SpatialMaxPooling.forward_c = inline.load [[
  const void* torch_Tensor_id = luaT_checktypename2id(L, "torch.Tensor");
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor_id);
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_Tensor_id);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor_id);

  luaL_argcheck(L, input->nDimension == 3, 2, "3D tensor expected");
  luaL_argcheck(L, input->size[0] >= kW && input->size[1] >= kH, 2, "input image smaller than kernel size");

  THTensor *outputPlane, *inputPlane, *unfoldedInputPlane, *localInput;
  int k,i,j;

  THTensor_resize3d(output,
                    (input->size[0] - kW) / dW + 1, 
                    (input->size[1] - kH) / dH + 1,
                    input->size[2]);

  inputPlane = THTensor_new();
  outputPlane = THTensor_new();
  localInput = THTensor_new();
  unfoldedInputPlane = THTensor_new();


  /* indices will contain i,j locatyions for each output point */
  THTensor_resize4d(indices, output->size[0],output->size[1],output->size[2],2);

  for (k = 0; k < input->size[2]; k++)
  {
    /* get input and output plane */
    THTensor_select(outputPlane, output, 2, k);
    THTensor_select(inputPlane, input, 2, k);

    /* Unfold input to get each local window */
    THTensor_unfold(unfoldedInputPlane, inputPlane, 0, kW, dW);
    THTensor_unfold(unfoldedInputPlane, NULL,       1, kH, dH);

    /* Calculate max points */
    for(j = 0; j < outputPlane->size[1]; j++) {
      for(i = 0; i < outputPlane->size[0]; i++) {
	long maxindex = -1;
	double maxval = -THInf;
	long tcntr = 0;
        int x,y;
        for(y = 0; y < unfoldedInputPlane->size[3]; y++) {
          for(x = 0; x < unfoldedInputPlane->size[2]; x++) {
            double val = THTensor_get4d(unfoldedInputPlane, i,j,x,y);
            if (val > maxval) {
              maxval = val;
              maxindex = tcntr;
            }
            tcntr++;
          }
        }

	THTensor_set4d(indices,i,j,k,0, (maxindex %% dW) +1);
	THTensor_set4d(indices,i,j,k,1, (int)(maxindex / dW)+1);
	THTensor_set2d(outputPlane,i,j,maxval);
      }
    }
  }
  THTensor_free(inputPlane);
  THTensor_free(outputPlane);
  THTensor_free(unfoldedInputPlane);
  THTensor_free(localInput);

  return 1;
]]

inline.headers('omp.h')
inline.flags('-fopenmp')
SpatialMaxPooling.forward_c_omp = inline.load [[
  const void* torch_Tensor_id = luaT_checktypename2id(L, "torch.Tensor");
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor_id);
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_Tensor_id);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor_id);

  luaL_argcheck(L, input->nDimension == 3, 2, "3D tensor expected");
  luaL_argcheck(L, input->size[0] >= kW && input->size[1] >= kH, 2, "input image smaller than kernel size");

  THTensor_resize3d(output,
                    (input->size[0] - kW) / dW + 1, 
                    (input->size[1] - kH) / dH + 1,
                    input->size[2]);

  /* indices will contain i,j locatyions for each output point */
  THTensor_resize4d(indices, output->size[0],output->size[1],output->size[2],2);

  omp_lock_t lock_input,lock_output; 
  omp_init_lock(&lock_input); omp_init_lock(&lock_output);

  int k,i,j;
  #pragma omp parallel for private(i,j,k)
  for (k = 0; k < input->size[2]; k++)
  {
    THTensor *outputPlane, *inputPlane, *unfoldedInputPlane, *localInput;
    omp_set_lock(&lock_input);
    inputPlane = THTensor_new();
    THTensor_select(inputPlane, input, 2, k);
    unfoldedInputPlane = THTensor_new();
    THTensor_unfold(unfoldedInputPlane, inputPlane, 0, kW, dW);
    THTensor_unfold(unfoldedInputPlane, NULL,       1, kH, dH);
    localInput = THTensor_new();
    omp_unset_lock(&lock_input);

    omp_set_lock(&lock_output);
    outputPlane = THTensor_new();
    THTensor_select(outputPlane, output, 2, k);
    omp_unset_lock(&lock_output);

    /* Calculate max points */
    for(j = 0; j < outputPlane->size[1]; j++) {
      for(i = 0; i < outputPlane->size[0]; i++) {
	long maxindex = -1;
	double maxval = -THInf;
	long tcntr = 0;
        int x,y;
        for(y = 0; y < unfoldedInputPlane->size[3]; y++) {
          for(x = 0; x < unfoldedInputPlane->size[2]; x++) {
            double val = THTensor_get4d(unfoldedInputPlane, i,j,x,y);
            if (val > maxval) {
              maxval = val;
              maxindex = tcntr;
            }
            tcntr++;
          }
        }

	THTensor_set4d(indices,i,j,k,0, (maxindex %% dW) +1);
	THTensor_set4d(indices,i,j,k,1, (int)(maxindex / dW)+1);
	THTensor_set2d(outputPlane,i,j,maxval);
      }
    }

    //#pragma omp barrier
    omp_set_lock(&lock_input);
    THTensor_free(inputPlane);
    THTensor_free(unfoldedInputPlane);
    THTensor_free(localInput);
    omp_unset_lock(&lock_input);

    omp_set_lock(&lock_output);
    THTensor_free(outputPlane);
    omp_unset_lock(&lock_output);
  }

  omp_destroy_lock(&lock_input); omp_destroy_lock(&lock_output);
  return 1;
]]

SpatialMaxPooling.backward_c = inline.load [[
  const void* torch_Tensor_id = luaT_checktypename2id(L, "torch.Tensor");
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor_id);  
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor_id);  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_Tensor_id);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor_id);

  THTensor *gradOutputPlane, *gradInputPlane, *unfoldedGradInputPlane, *gradLocalInput;
  int k,i,j;

  THTensor_resizeAs(gradInput, input);
  THTensor_zero(gradInput);

  gradInputPlane = THTensor_new();
  gradOutputPlane = THTensor_new();
  gradLocalInput = THTensor_new();
  unfoldedGradInputPlane = THTensor_new();

  for (k = 0; k < input->size[2]; k++)
  {
    /* get input and output plane */
    THTensor_select(gradOutputPlane, gradOutput, 2, k);
    THTensor_select(gradInputPlane, gradInput, 2, k);

    /* Unfold input to get each local window */
    THTensor_unfold(unfoldedGradInputPlane, gradInputPlane, 0, kW, dW);
    THTensor_unfold(unfoldedGradInputPlane, NULL,           1, kH, dH);

    /* Calculate max points */
    for(i = 0; i < gradOutputPlane->size[0]; i++)
    {
      for(j = 0; j < gradOutputPlane->size[1]; j++)
      {
	THTensor_select(gradLocalInput, unfoldedGradInputPlane,0,i);
	THTensor_select(gradLocalInput, NULL,                  0,j);
	long maxi = THTensor_get4d(indices,i,j,k,0)-1;
	long maxj = THTensor_get4d(indices,i,j,k,1)-1;
	double gi = THTensor_get2d(gradLocalInput,maxi,maxj)+
	  THTensor_get2d(gradOutputPlane,i,j);
	THTensor_set2d(gradLocalInput,maxi,maxj,gi);
      }
    }
  }

  THTensor_free(gradInputPlane);
  THTensor_free(gradOutputPlane);
  THTensor_free(unfoldedGradInputPlane);
  THTensor_free(gradLocalInput);

  return 1;
]]

function SpatialMaxPooling:forward(input)
   if openmp and openmp.enabled then
      self:forward_c_omp(input)
   else
      self:forward_c(input)
   end
   return self.output
end

function SpatialMaxPooling:backward(input, gradOutput)
   self:backward_c(input, gradOutput)
   return self.gradInput
end

function SpatialMaxPooling:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
   self.indices:resize()
   self.indices:storage():resize(0)
end

function SpatialMaxPooling:write(file)
   parent.write(self, file)
   file:writeInt(self.kW)
   file:writeInt(self.kH)
   file:writeInt(self.dW)
   file:writeInt(self.dH)
   file:writeObject(self.indices)
end

function SpatialMaxPooling:read(file)
   parent.read(self, file)
   self.kW = file:readInt()
   self.kH = file:readInt()
   self.dW = file:readInt()
   self.dH = file:readInt()
   self.indices = file:readObject()
end
