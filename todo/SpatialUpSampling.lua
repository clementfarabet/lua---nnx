local SpatialUpSampling, parent = torch.class('nn.SpatialUpSampling', 'nn.Module')

local help_desc =
   [[Applies a 2D up-sampling over an input image composed of
      several input planes. The input tensor in forward(input) is
      expected to be a 3D tensor (width x height x nInputPlane).
      The number of output planes will be the same as nInputPlane.

      If the input image is a 3D tensor width x height x nInputPlane,
      the output image size will be owidth x oheight x nInputPlane where

      owidth  = width*dW
      oheight  = height*dH ]]

function SpatialUpSampling:__init(...)
   parent.__init(self)

   -- get args
   xlua.unpack_class(self, {...}, 'nn.SpatialUpSampling',  help_desc,
                        {arg='nInputPlane', type='number', help='number of input planes', req=true},
                        {arg='dW', type='number', help='stride width', req=true},
                        {arg='dH', type='number', help='stride height', req=true})
end

-- needed for Recursive fovea
function SpatialUpSampling:configure(np, dW,dH)
   self.nInputPlane = np
   self.dW = dW
   self.dH = dH
end
   
-- define fprop in C
SpatialUpSampling.forward_c = inline.load [[
      // get all params
      const void* torch_Tensor_id = luaT_checktypename2id(L, "torch.Tensor");
      THTensor *input = luaT_checkudata(L, 2, torch_Tensor_id);
      int dW = luaT_getfieldcheckint(L, 1, "dW");
      int dH = luaT_getfieldcheckint(L, 1, "dH");
      THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor_id);

      // dims
      int iwidth = input->size[0];
      int iheight = input->size[1];
      int ochannels = input->size[2];
      int owidth = iwidth * dW;
      int oheight = iheight * dH;

      // select planes
      THTensor *outputPlane = THTensor_new();
      THTensor *inputPlane = THTensor_new();

      // resample each plane
      int k;
      for (k=0; k<ochannels; k++) {
         // get planes
         THTensor_select(inputPlane, input, 2, k);
         THTensor_select(outputPlane, output, 2, k);

         // for each plane, resample
         int x,y;
         for (y=0; y<oheight; y++) {
            for (x=0; x<owidth; x++) {
               // input positions (floored)
               int ix = x/dW;
               int iy = y/dH;

               // set output
               THTensor_set2d(outputPlane, x, y, THTensor_get2d(inputPlane, ix, iy));
            }
         }
      }
      // cleanup
      THTensor_free(inputPlane);
      THTensor_free(outputPlane);
      return 1;
]]

-- define bprop in C
SpatialUpSampling.backward_c = inline.load [[
      // get all params
      const void* torch_Tensor_id = luaT_checktypename2id(L, "torch.Tensor");
      THTensor *input = luaT_checkudata(L, 2, torch_Tensor_id);
      THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor_id);
      THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor_id);
      int dW = luaT_getfieldcheckint(L, 1, "dW");
      int dH = luaT_getfieldcheckint(L, 1, "dH");

      // dims
      int iwidth = input->size[0];
      int iheight = input->size[1];
      int ichannels = input->size[2];
      int owidth = gradOutput->size[0];
      int oheight = gradOutput->size[1];
      int ochannels = gradOutput->size[2];

      // resize gradInput
      THTensor_zero(gradInput);

      // select planes
      THTensor *gradOutputPlane = THTensor_new();
      THTensor *gradInputPlane = THTensor_new();

      // compute gradients for each plane
      int k;
      for (k=0; k<ochannels; k++) {
         // get planes
         THTensor_select(gradInputPlane, gradInput, 2, k);
         THTensor_select(gradOutputPlane, gradOutput, 2, k);

         // for each plane, resample
         int x,y;
         for (y=0; y<oheight; y++) {
            for (x=0; x<owidth; x++) {
               // input positions (floored)
               int ix = x/dW;
               int iy = y/dH;

               // output gradient
               double ograd = THTensor_get2d(gradOutputPlane, x, y);

               // accumulate gradient
               THTensor_set2d(gradInputPlane, ix, iy, THTensor_get2d(gradInputPlane, ix, iy) + ograd);
            }
         }
      }

      // cleanup
      THTensor_free(gradInputPlane);
      THTensor_free(gradOutputPlane);
      return 1;
]]

function SpatialUpSampling:forward(input)
   self.output:resize(input:size(1) * self.dW, input:size(2) * self.dH, input:size(3))
   self:forward_c(input)
   return self.output
end

function SpatialUpSampling:backward(input, gradOutput)
   self.gradInput:resize(input:size(1), input:size(2), input:size(3))
   self:backward_c(input, gradOutput)
   return self.gradInput
end

function SpatialUpSampling:write(file)
   parent.write(self, file)
   file:writeInt(self.dW)
   file:writeInt(self.dH)
   file:writeInt(self.nInputPlane)
end

function SpatialUpSampling:read(file)
   parent.read(self, file)
   self.dW = file:readInt()
   self.dH = file:readInt()
   self.nInputPlane = file:readInt()
end
