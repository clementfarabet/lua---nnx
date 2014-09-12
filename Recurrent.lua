------------------------------------------------------------------------
--[[ Recurrent ]]--
-- Ref.: http://goo.gl/vtVGkO (Mikolov et al.)
-- Simple Recurrent Neural Network (or Elman Network)
-- input is batchSize x nFrame x frameSize
-- TODO : 
-- Recurrent: works with a LookupTable (2D inputs can vary, 3D can't)
-- RecurrentTable: works with a list of variable sized 2D examples
-- RecurrentContainer: takes an input and feedback Module with table inputs
------------------------------------------------------------------------
local Recurrent, parent = torch.class('nn.Recurrent', 'nn.Module')

function Recurrent:__init(outputSize, transfer, initialValue)
   parent.__init(self)

   self.outputSize = outputSize
   self.transfer = transfer or nn.Sigmoid()
   self.weight = torch.Tensor(outputSize, outputSize)
   self.bias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(outputSize, outputSize)
   self.gradBias = torch.Tensor(outputSize)
   self.initialValue = initialValue or 0.1
   self.initialState = torch.Tensor()
   self.states = {self.initialState}
   
   self:reset()
end

function Recurrent:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   self.weight:uniform(-stdv, stdv)
   self.bias:uniform(-stdv, stdv)
end

function Recurrent:updateOutput(input)
   if input:dim() == 2 then
      self.output:resize(self.bias:size(1))
      self.output:copy(self.bias)
      self.output:addmv(1, self.weight, input)
   elseif input:dim() == 3 then
      local nframe = input:size(1)
      local nunit = self.bias:size(1)
      self.initialState:resize(nframe, nunit):fill(self.initialValue)
      for i=1,input:size(2) do
         local output = self.states[i+1] or self.initialState:clone()
         local input = input:select(2,i)
         output:copy(self.states[i])
         if nunit == 1 then
            -- Special case to fix output size of 1 bug:
            output:add(self.bias[1])
            output:select(2,1):addmv(1, input, self.weight:select(1,1))
         else
            output:addr(1, input.new(nframe):fill(1), self.bias)
            output:addmm(1, input, self.weight:t())
         end
      end
   else
      error('input must be vector or matrix')
   end

   return self.output
end

function Recurrent:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
      elseif input:dim() == 2 then
         self.gradInput:addmm(0, 1, gradOutput, self.weight)
      end

      return self.gradInput
   end
end

function Recurrent:accGradParameters(input, gradOutput, scale)
   scale = scale or 1

   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
      self.gradBias:add(scale, gradOutput)      
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nunit = self.bias:size(1)
      
      if nunit == 1 then
         -- Special case to fix output size of 1 bug:
         self.gradWeight:select(1,1):addmv(scale, input:t(), gradOutput:select(2,1))
         self.gradBias:addmv(scale, gradOutput:t(), input.new(nframe):fill(1))
      else
         self.gradWeight:addmm(scale, gradOutput:t(), input)
         self.gradBias:addmv(scale, gradOutput:t(), input.new(nframe):fill(1))
      end
   end

end
