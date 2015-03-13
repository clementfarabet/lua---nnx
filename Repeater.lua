------------------------------------------------------------------------
--[[ Repeater ]]--
-- Encapsulates an AbstractRecurrent instance (rnn) which is repeatedly 
-- presented with the same input for nStep time steps.
-- The output is a table of nStep outputs of the rnn.
------------------------------------------------------------------------
local Repeater, parent = torch.class("nn.Repeater", "nn.Container")

function Repeater:__init(nStep, rnn)
   parent.__init(self)
   self.nStep = nStep
   self.rnn = rnn
   assert(rnn.backwardThroughTime, "expecting AbstractRecurrent instance for arg 2")
   self.modules[1] = rnn
   self.output = {}
end

function Repeater:updateOutput(input)
   self.rnn:forget()
   for step=1,self.nStep do
      self.output[step] = self.rnn:updateOutput(input)
   end
   return self.output
end

function Repeater:updateGradInput(input, gradOutput)
   assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
   for step=1,self.nStep do
      self.rnn.step = step + 1
      self.rnn:updateGradInput(input, gradOutput[step])
   end
   -- back-propagate through time (BPTT)
   self.rnn:updateGradInputThroughTime()
   self.gradInput = self.rnn.gradInputs
   return self.gradInput
end

function Repeater:accGradParameters(input, gradOutput, scale)
   assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
   for step=1,self.nStep do
      self.rnn.step = step + 1
      self.rnn:accGradParameters(input, gradOutput[step], scale)
   end
   -- back-propagate through time (BPTT)
   self.rnn:accGradParametersThroughTime()
end

function Repeater:accUpdateGradParameters(input, gradOutput, lr)
   assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
   for step=1,self.nStep do
      self.rnn.step = step + 1
      self.rnn:accGradParameters(input, gradOutput[step], 1)
   end
   -- back-propagate through time (BPTT)
   self.rnn:accUpdateGradParametersThroughTime(lr)
end
