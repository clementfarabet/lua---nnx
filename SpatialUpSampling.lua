local SpatialUpSampling, parent = torch.class('nn.SpatialUpSampling', 'nn.Module')

local help_desc = [[
Applies a 2D up-sampling over an input image composed of
several input planes. The input tensor in forward(input) is
expected to be a 3D tensor (width x height x nInputPlane).
The number of output planes will be the same as nInputPlane.

The upsampling is done using the simple nearest neighbor
technique. For interpolated (bicubic) upsampling, use 
nn.SpatialReSampling().

If the input image is a 3D tensor width x height x nInputPlane,
the output image size will be owidth x oheight x nInputPlane where

owidth  = width*dW
oheight  = height*dH ]]

function SpatialUpSampling:__init(...)
   parent.__init(self)

   -- get args
   xlua.unpack_class(self, {...}, 'nn.SpatialUpSampling',  help_desc,
                     {arg='dW', type='number', help='stride width', req=true},
                     {arg='dH', type='number', help='stride height', req=true})
end

function SpatialUpSampling:forward(input)
   self.output:resize(input:size(1), input:size(2) * self.dH, input:size(3) * self.dW)
   input.nn.SpatialUpSampling_forward(self, input)
   return self.output
end

function SpatialUpSampling:backward(input, gradOutput)
   self.gradInput:resizeAs(input)
   input.nn.SpatialUpSampling_backward(self, input, gradOutput)
   return self.gradInput
end

function SpatialUpSampling:write(file)
   parent.write(self, file)
   file:writeInt(self.dW)
   file:writeInt(self.dH)
end

function SpatialUpSampling:read(file)
   parent.read(self, file)
   self.dW = file:readInt()
   self.dH = file:readInt()
end
