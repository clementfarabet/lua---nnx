local SpatialMatching, parent = torch.class('nn.SpatialMatching', 'nn.Module')

function SpatialMatching:__init(maxw, maxh, full_output)
   -- If full_output is false, output is computed on elements of the first input
   -- for which all the possible corresponding elements exist in the second input
   -- In addition, if full_output is set to false, the pixel (1,1) of the first input
   -- is supposed to correspond to the pixel (maxh/2, maxw/2) of the second one
   -- If align_inputs is set, input[2] is assumed to be large enough
   -- (that is input[1]:size(1) <= inputs[1]:size(1) - maxh + 1, same for w)
   -- TODO full_output == true and align_inputs == false is probably useless
   parent.__init(self)
   self.maxw = maxw or 11
   self.maxh = maxh or 11
   full_output = full_output or true
   if full_output then self.full_output = 1 else self.full_output = 0 end
end

function SpatialMatching:updateOutput(input)
   -- input is a table of 2 inputs, each one being KxHxW
   -- if not full_output, the 1st one is KxH1xW1 where H1 <= H-maxh+1, W1 <= W-maxw+1
   self.output:resize(input[1]:size(2), input[1]:size(3), self.maxh, self.maxw)
   input[1].nn.SpatialMatching_updateOutput(self, input[1], input[2])
   return self.output
end

function SpatialMatching:updateGradInput(input, gradOutput)
   -- TODO this is probably the wrong way
   self.gradInput1 = torch.Tensor(input[1]:size()):zero()
   self.gradInput2 = torch.Tensor(input[2]:size()):zero()
   input[1].nn.SpatialMatching_updateGradInput(self, input[1], input[2], gradOutput)
   self.gradInput = {self.gradInput1, self.gradInput2}
   return self.gradInput
end
