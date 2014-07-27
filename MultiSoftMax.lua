------------------------------------------------------------------------
--[[ MultiSoftMax ]]--
-- Takes 2D or 3D input and performs a softmax over the last dimension.
------------------------------------------------------------------------
local MultiSoftMax, parent = torch.class('nn.MultiSoftMax', 'nn.Module')

function MultiSoftMax.__init(self)
   parent.__init(self)
   self._input = torch.Tensor()
   self._output = torch.Tensor()
   self._gradInput = torch.Tensor()
   self._gradOutput = torch.Tensor()
end

function MultiSoftMax:updateOutput(input)
   if input:dim() == 2 then
      return input.nn.SoftMax_updateOutput(self, input)
   end
   if input:dim() ~= 3 then
      error"Only supports 2D or 3D inputs"
   end
   self._input:view(input, input:size(1)*input:size(2), input:size(3))
   local output = self.output
   self.output = self._output
   input.nn.SoftMax_updateOutput(self, self._input)
   output:viewAs(self.output, input)
   self.output = output
   return self.output
end

function MultiSoftMax:updateGradInput(input, gradOutput)
   if input:dim() == 2 then
      return input.nn.SoftMax_updateGradInput(self, input, gradOutput)
   end
   self._gradOutput:view(gradOutput, input:size(1)*input:size(2), input:size(3))
   local gradInput = self.gradInput
   self.gradInput = self._gradInput
   local output = self.output
   self.output = self._output
   input.nn.SoftMax_updateGradInput(self, self._input, self._gradOutput)
   self.gradInput = gradInput:viewAs(self.gradInput, input)
   self.output = output
   return self.gradInput
end
