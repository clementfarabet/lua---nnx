

local CAddTable, parent = torch.class('nn.CAddTable', 'nn.Module')

function CAddTable:__init()
   parent.__init(self)
   self.gradInput = {}
end

function CAddTable:forward(input)
   
   self.output:resizeAs(input[1]):copy(input[1])

   -- sum everything
   for i=2,#input do
      self.output:add(input[i])
   end
   return self.output
end

function CAddTable:backward(input, gradOutput)
   
   for i=1,#input do
      if self.gradInput[i] == nil then
	 self.gradInput[i] = torch.Tensor()
      end
      self.gradInput[i]:resizeAs(input[i])
      self.gradInput[i]:copy(gradOutput)
   end
   return self.gradInput
end


function CAddTable:write(file)
   parent.write(self, file)
end

function CAddTable:read(file)
   parent.read(self, file)
end
