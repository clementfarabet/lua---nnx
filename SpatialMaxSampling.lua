local SpatialMaxSampling, parent = torch.class('nn.SpatialMaxSampling', 'nn.Module')

function SpatialMaxSampling:__init(...)
   parent.__init(self)
   xlua.unpack_class(
      self, {...}, 'nn.SpatialMaxSampling',
      'resample an image using max selection',
      {arg='rwidth', type='number', help='ratio: owidth/iwidth'},
      {arg='rheight', type='number', help='ratio: oheight/iheight'},
      {arg='owidth', type='number', help='output width'},
      {arg='oheight', type='number', help='output height'}
   )
   self.indices = torch.Tensor()
end

function SpatialMaxSampling:forward(input)
   self.oheight = self.oheight or self.rheight*input:size(2)
   self.owidth = self.owidth or self.rwidth*input:size(3)
   input.nn.SpatialMaxSampling_forward(self, input)
   return self.output
end

function SpatialMaxSampling:backward(input, gradOutput)
   input.nn.SpatialMaxSampling_backward(self, input, gradOutput)
   return self.gradInput
end
