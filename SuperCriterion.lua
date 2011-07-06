local SuperCriterion, parent = torch.class('nn.SuperCriterion', 'nn.Criterion')

function SuperCriterion:__init()
   parent.__init(self)
   self.criterions = {}
   self.weights = {}
   self.gradInput = {}
end

function SuperCriterion:add(criterion, weight)
   weight = weight or 1
   table.insert(self.criterions, criterion)
   table.insert(self.weights, weight)
end

function SuperCriterion:forward(input, target)
   self.output = 0
   for i,criterion in ipairs(self.criterions) do
      self.output = self.output + self.weights[i]*criterion:forward(input[i],target)
   end
   return self.output
end

function SuperCriterion:backward(input, target)
   for i,criterion in ipairs(self.criterions) do
      self.gradInput[i] = torch.Tensor() or self.gradInput[i]
      self.gradInput[i]:resizeAs(input[i]):zero()
      self.gradInput[i]:add(self.weights[i], criterion:backward(input[i],target) )
   end
   return self.gradInput
end
