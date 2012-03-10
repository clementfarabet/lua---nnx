local SpatialMatching, parent = torch.class('nn.SpatialMatching', 'nn.Module')

function SpatialMatching:__init(maxw, maxh, full_output)
   parent.__init(self)
   self.maxw = maxw or 11
   self.maxh = maxh or 11
   full_output = full_output or false
   if full_output then
      self.full_output = 1
   else
      self.full_output = 0
   end
end

function SpatialMatching:updateOutput(input)
   -- input is a table of 2 inputs, each one being KxHxW
   if self.full_output == 1 then
      self.output:resize(input[1]:size(2), input[1]:size(3), self.maxh, self.maxw)
   else
      self.output:resize(input[1]:size(2)-self.maxh+1, input[1]:size(3)-self.maxw+1,
			 self.maxh, self.maxw)
   end
   input[1].nn.SpatialMatching_updateOutput(self, input[1], input[2])
   return self.output
end

function SpatialMatching:updateGradInput(input, gradOutput)
   -- todo this is probably wrong
   self.gradInput1 = torch.Tensor():resizeAs(input[1]):zero()
   self.gradInput2 = torch.Tensor():resizeAs(input[2]):zero()
   input[1].nn.SpatialMatching_updateGradInput(self, input[1], input[2], gradOutput)
   self.gradInput = {self.gradInput1, self.gradInput2}
   return self.gradInput
end
