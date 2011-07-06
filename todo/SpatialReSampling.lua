local SpatialReSampling, parent = torch.class('nn.SpatialReSampling', 'nn.Module')

local help_desc =
[[Applies a 2D re-sampling over an input image composed of
several input planes. The input tensor in forward(input) is 
expected to be a 3D tensor (width x height x nInputPlane). 
The number of output planes will be the same as the nb of input
planes.

If the input image is a 3D tensor width x height x nInputPlane, 
the output image size will be owidth x oheight x nInputPlane where
owidth and oheight are given to the constructor. ]]

function SpatialReSampling:__init(...)
   parent.__init(self)
   xlua.unpack_class(
      self, {...}, 'nn.SpatialReSampling', help_desc,
      {arg='owidth', type='number', help='output width', req=true},
      {arg='oheight', type='number', help='output height', req=true}
   )
end

function SpatialReSampling:write(file)
   parent.write(self, file)
   file:writeInt(self.owidth)
   file:writeInt(self.oheight)
   -- file:writeString(self.mode)
end

function SpatialReSampling:read(file)
   parent.read(self, file)
   self.owidth = file:readInt()
   self.oheight = file:readInt()
   -- self.mode = file:readString()
end
