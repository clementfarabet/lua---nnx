local SpatialMatching, parent = torch.class('nn.SpatialMatching', 'nn.Module')

function SpatialMatching:__init(maxw, maxh)
   parent.__init(self)
   self.maxw = maxw or 11
   self.maxh = maxh or 11
end

function SpatialMatching:updateOutput(input)
   -- input is a table of 2 inputs, each one being KxHxW
   self.output:resize(input[1]:size(2), input[1]:size(3), self.maxh, self.maxw)
   input[1].nn.SpatialMatching_updateOutput(self, input[1], input[2])
   return self.output
end

function SpatialMatching:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   input[1].nn.SpatialMatching_updateGradInput(self, input[1], input[2], gradOutput)
   return self.gradInput
end
