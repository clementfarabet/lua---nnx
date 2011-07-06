local SpatialLogSoftMax, parent = torch.class('nn.SpatialLogSoftMax', 'nn.Module')

function SpatialLogSoftMax:__init()
   parent.__init(self)
end

function SpatialLogSoftMax:forward(input)
   self.output:resizeAs(input)
   input.nn.SpatialLogSoftMax_forward(self, input)
   return self.output
end

function SpatialLogSoftMax:backward(input, gradOutput)
   self.gradInput:resizeAs(input)
   input.nn.SpatialLogSoftMax_backward(self, input, gradOutput)
   return self.gradInput
end
