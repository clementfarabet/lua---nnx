local SpatialMaxPooling, parent = torch.class('nn.SpatialMaxPooling', 'nn.Module')

local help_desc =
[[Applies a 2D sub-sampling over an input image composed of
several input planes. The input tensor in forward(input) is 
expected to be a 3D tensor (nInputPlane x height x width). 
The number of output planes will be the same as nInputPlane.

Compared to the nn.SpatialSubSampling module, a max operator is
used to pool values in the kHxkW input neighborhood.

Note that depending of the size of your kernel, several 
(of the last) columns or rows of the input image might be lost. 
It is up to the user to add proper padding in images.

If the input image is a 3D tensor nInputPlane x height x width,
the output image size will be nInputPlane x oheight x owidth, where

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

function SpatialMaxPooling:forward(input)
   input.nn.SpatialMaxPooling_forward(self, input)
   return self.output
end

function SpatialMaxPooling:backward(input, gradOutput)
   input.nn.SpatialMaxPooling_backward(self, input, gradOutput)
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
