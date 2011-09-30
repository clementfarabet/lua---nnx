local SpatialMaxSampling, parent = torch.class('nn.SpatialMaxSampling', 'nn.Module')

function SpatialMaxSampling:__init(...)
   parent.__init(self)
   xlua.unpack_class(
      self, {...}, 'nn.SpatialMaxSampling',
      'resample an image using max selection',
      {arg='owidth', type='number', help='output width'},
      {arg='oheight', type='number', help='output height'}
   )
   self.indices = torch.Tensor()
end

function SpatialMaxSampling:forward(input)
   input.nn.SpatialMaxSampling_forward(self, input)
   return self.output
end

function SpatialMaxSampling:backward(input, gradOutput)
   input.nn.SpatialMaxSampling_backward(self, input, gradOutput)
   return self.gradInput
end
