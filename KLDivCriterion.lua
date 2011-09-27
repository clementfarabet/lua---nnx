local KLDivCriterion, parent = torch.class('nn.KLDivCriterion', 'nn.Criterion')

function KLDivCriterion:__init()
   parent.__init(self)
   -- user options
   self.inputIsProbability = false
   self.targetIsProbability = false
   -- internal
   self.targetSoftMax = nn.SoftMax()
   self.inputSoftMax = nn.SoftMax()
   self.gradProbInput = torch.Tensor()
end

function KLDivCriterion:normalize(input, target)
   -- normalize target
   if not self.targetIsProbability then
      self.probTarget = self.targetSoftMax:forward(target)
   else
      self.probTarget = target
   end

   -- normalize input
   if not self.inputIsProbability then
      self.probInput = self.inputSoftMax:forward(input)
   else
      self.probInput = input
   end
end

function KLDivCriterion:denormalize(input)
   -- denormalize gradients
   if not self.inputIsProbability then
      self.gradInput = self.inputSoftMax:backward(input, self.gradProbInput)
   else
      self.gradInput = self.gradProbInput
   end
end

function KLDivCriterion:forward(input, target)
   self:normalize(input, target)
   self.output = 0
   for i = 1,input:size(1) do
      local acc = self.probTarget[i] * math.log(math.max(self.probTarget[i],1e-9) / math.max(self.probInput[i],1e-9))
      self.output = self.output + acc
   end
   return self.output
end

function KLDivCriterion:backward(input, target)
   self:normalize(input, target)
   self.gradProbInput:resizeAs(input)
   for i = 1,input:size(1) do
      self.gradProbInput[i] = - self.probTarget[i] / math.max(self.probInput[i],1e-9)
   end
   self:denormalize(input)
   return self.gradInput
end
