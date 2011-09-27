local DistMarginCriterion, parent = torch.class('nn.DistMarginCriterion', 'nn.Criterion')

function DistMarginCriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function DistMarginCriterion:forward(input, target)
   return input.nn.DistMarginCriterion_forward(self, input, target)
end

function DistMarginCriterion:backward(input, target)
   return input.nn.DistMarginCriterion_backward(self, input, target)
end
