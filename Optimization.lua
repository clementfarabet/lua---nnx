local Optimization = torch.class('nn.Optimization')

function Optimization:__init()
end

function Optimization:forward(parameters, gradParameters)
   self:flatten(parameters, gradParameters)
   -- do your thing
   self:unflatten(parameters, gradParameters)
end

function Optimization:flatten(parameters, gradParameters)
   if type(parameters) == 'table' then
      -- create flat parameters
      self.parameters = self.parameters or torch.Tensor()
      self.gradParameters = self.gradParameters or torch.Tensor()
      -- assuming that the parameters won't change their size, 
      -- we compute offsets once
      if not self.offsets then
         self.nParameters = 0
         self.offsets = {}
         for _,param in ipairs(parameters) do
            table.insert(self.offsets, self.nParameters+1)
            self.nParameters = self.nParameters + param:nElement()
         end
         self.parameters:resize(self.nParameters)
         self.gradParameters:resize(self.nParameters)
      end
      -- copy all params in flat array
      for i = 1,#parameters do
         local nElement = parameters[i]:nElement()
         self.parameters:narrow(1,self.offsets[i],nElement):copy(parameters[i])
         self.gradParameters:narrow(1,self.offsets[i],nElement):copy(gradParameters[i])
      end
   else
      self.parameters = parameters
      self.gradParameters = gradParameters
   end
end

function Optimization:unflatten(parameters, gradParameters)
   if type(parameters) == 'table' then
      -- copy all params into unflat arrays
      local offset = 1
      for i = 1,#parameters do
         local nElement = parameters[i]:nElement()
         parameters[i]:copy(self.parameters:narrow(1,offset,nElement))
         gradParameters[i]:copy(self.gradParameters:narrow(1,offset,nElement))
         offset = offset + nElement
      end
   else
      parameters = self.parameters
      gradParameters = self.gradParameters
   end
end
