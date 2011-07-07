

local CMulTable, parent = torch.class('nn.CMulTable', 'nn.Module')

function CMulTable:__init()
   parent.__init(self)
   self.gradInput = {}
end

function CMulTable:forward(input)
   
   self.output:resizeAs(input[1]):copy(input[1])

   -- multiply everything
   for i=2,#input do
      self.output:cmul(input[i])
   end
   return self.output
end

function CMulTable:backward(input, gradOutput)
   
   local tout = torch.Tensor():resizeAs(self.output)
   for i=1,#input do
      if self.gradInput[i] == nil then
	 self.gradInput[i] = torch.Tensor()
      end
      self.gradInput[i]:resizeAs(input[i]):copy(gradOutput)
      tout:copy(self.output):cdiv(input[i])
      self.gradInput[i]:cmul(tout)
   end
   return self.gradInput
end


function CMulTable:write(file)
   parent.write(self, file)
end

function CMulTable:read(file)
   parent.read(self, file)
end
