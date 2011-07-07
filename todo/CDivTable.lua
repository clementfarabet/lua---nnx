

local CDivTable, parent = torch.class('nn.CDivTable', 'nn.Module')

function CDivTable:__init()
   parent.__init(self)
   self.gradInput = {}
   self.gradInput[1] = torch.Tensor()
   self.gradInput[2] = torch.Tensor()
end

function CDivTable:forward(input)
   
   self.output:resizeAs(input[1]):copy(input[1])
   self.output:cdiv(input[2])
   return self.output
end

function CDivTable:backward(input, gradOutput)
   self.gradInput[1]:resizeAs(input[1]):copy(gradOutput):cdiv(input[2])
   self.gradInput[2]:resizeAs(input[2]):zero():addcdiv(-1,self.gradInput[1],input[2]):cmul(input[1])
   return self.gradInput
end

function CDivTable:empty()
   self.gradInput[1]:resize()
   self.gradInput[1]:storage():resize(0)
   self.gradInput[2]:resize()
   self.gradInput[2]:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
end

function CDivTable:write(file)
   parent.write(self, file)
end

function CDivTable:read(file)
   parent.read(self, file)
end
