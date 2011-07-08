local SparseCriterion, parent = torch.class('nn.SparseCriterion', 'nn.Criterion')

function SparseCriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function SparseCriterion:forward(input)
   input.nn.SparseCriterion_forward(self, input)
   return self.output
end

function SparseCriterion:backward(input)
   input.nn.SparseCriterion_backward(self, input)
   return self.gradInput
end

function SparseCriterion:write(file)
   parent.write(self, file)
   file:writeBool(self.sizeAverage)
end

function SparseCriterion:read(file)
   parent.read(self, file)
   self.sizeAverage = file:readBool()
end
