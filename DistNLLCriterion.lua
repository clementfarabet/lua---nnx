local DistNLLCriterion, parent = torch.class('nn.DistNLLCriterion', 'nn.Criterion')

function DistNLLCriterion:__init()
   parent.__init(self)
   -- user options
   self.inputIsADistance = false
   self.inputIsProbability = false
   self.inputIsLogProbability = false
   self.targetIsProbability = false
   -- internal
   self.targetSoftMax = nn.SoftMax()
   self.inputLogSoftMax = nn.LogSoftMax()
   self.gradLogInput = torch.Tensor()
   self.input = torch.Tensor()
end

function DistNLLCriterion:normalize(input, target)
   -- normalize target
   if not self.targetIsProbability then
      self.probTarget = self.targetSoftMax:forward(target)
   else
      self.probTarget = target
   end

   -- flip input if a distance
   if self.inputIsADistance then
      self.input:resizeAs(input):copy(input):mul(-1)
   else
      self.input = input
   end

   -- normalize input
   if not self.inputIsLogProbability and not self.inputIsProbability then
      self.logProbInput = self.inputLogSoftMax:forward(self.input)
   elseif not self.inputIsLogProbability then
      print('TODO: implement nn.Log()')
   else
      self.logProbInput = self.input
   end
end

function DistNLLCriterion:denormalize(input)
   -- denormalize gradients
   if not self.inputIsLogProbability and not self.inputIsProbability then
      self.gradInput = self.inputLogSoftMax:backward(input, self.gradLogInput)
   elseif not self.inputIsLogProbability then
      print('TODO: implement nn.Log()')
   else
      self.gradInput = self.gradLogInput
   end

   -- if input is a distance, then flip gradients back
   if self.inputIsADistance then
      self.gradInput:mul(-1)
   end
end

function DistNLLCriterion:forward(input, target)
   self:normalize(input, target)
   self.output = 0
   for i = 1,input:size(1) do
      self.output = self.output - self.logProbInput[i] * self.probTarget[i]
   end
   return self.output
end

function DistNLLCriterion:backward(input, target)
   self:normalize(input, target)
   self.gradLogInput:resizeAs(input)
   for i = 1,input:size(1) do
      self.gradLogInput[i] = -self.probTarget[i]
   end
   self:denormalize(input)
   return self.gradInput
end

function DistNLLCriterion:write(file)
   parent.write(self, file)
   file:writeBool(self.inputIsProbability)
   file:writeBool(self.inputIsLogProbability)
   file:writeBool(self.targetIsProbability)
   file:writeObject(self.targetSoftMax)
   file:writeObject(self.inputLogSoftMax)
   file:writeObject(self.gradLogInput)
end

function DistNLLCriterion:read(file)
   parent.read(self, file)
   self.inputIsProbability = file:readBool()
   self.inputIsLogProbability = file:readBool()
   self.targetIsProbability = file:readBool()
   self.targetSoftMax = file:readObject()
   self.inputLogSoftMax = file:readObject()
   self.gradLogInput = file:readObject()
end
