local SparseCriterion, parent = torch.class('nn.SparseCriterion', 'nn.Criterion')

function SparseCriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function SparseCriterion:write(file)
   parent.write(self, file)
   file:writeBool(self.sizeAverage)
end

function SparseCriterion:read(file)
   parent.read(self, file)
   self.sizeAverage = file:readBool()
end
